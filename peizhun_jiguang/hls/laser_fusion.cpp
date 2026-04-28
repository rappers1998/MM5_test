#include "laser_fusion.hpp"

static pix8_t clip_u8(ap_int<32> value) {
#pragma HLS INLINE
    if (value < 0) {
        return 0;
    }
    if (value > 255) {
        return 255;
    }
    return (pix8_t)value;
}

void da1501a_parser_reset(da1501a_parser_t *parser) {
#pragma HLS INLINE
    parser->state = 0;
    parser->cmd = 0;
    parser->status = 0;
    parser->reserved = 0;
    parser->data_h = 0;
    parser->data_l = 0;
    parser->checksum_acc = 0;
}

bool da1501a_protocol1_parse_byte(
    ap_uint<8> byte_in,
    da1501a_parser_t *parser,
    ap_uint<32> *distance_mm
) {
#pragma HLS INLINE off
    bool done = false;
    *distance_mm = 0;

    switch (parser->state) {
    case 0:
        if (byte_in == 0x55) {
            parser->state = 1;
        }
        break;
    case 1:
        if (byte_in == 0xAA) {
            parser->state = 2;
            parser->checksum_acc = 0x55 + 0xAA;
        } else {
            da1501a_parser_reset(parser);
        }
        break;
    case 2:
        parser->cmd = byte_in;
        parser->checksum_acc += byte_in;
        parser->state = 3;
        break;
    case 3:
        parser->status = byte_in;
        parser->checksum_acc += byte_in;
        parser->state = 4;
        break;
    case 4:
        parser->reserved = byte_in;
        parser->checksum_acc += byte_in;
        parser->state = 5;
        break;
    case 5:
        parser->data_h = byte_in;
        parser->checksum_acc += byte_in;
        parser->state = 6;
        break;
    case 6:
        parser->data_l = byte_in;
        parser->checksum_acc += byte_in;
        parser->state = 7;
        break;
    case 7: {
        ap_uint<8> expected = (ap_uint<8>)(parser->checksum_acc & 0xFF);
        bool checksum_ok = (byte_in == expected);
        bool status_ok = (parser->status == 1);
        bool reserved_ok = (parser->reserved == 0xFF);
        bool data_ok = !((parser->data_h == 0xFF) && (parser->data_l == 0xFF));
        if (checksum_ok && status_ok && reserved_ok && data_ok) {
            ap_uint<16> decimeter = ((ap_uint<16>)parser->data_h << 8) | parser->data_l;
            ap_uint<32> decimeter32 = decimeter;
            *distance_mm = (decimeter32 << 6) + (decimeter32 << 5) + (decimeter32 << 2);
            done = true;
        }
        da1501a_parser_reset(parser);
        break;
    }
    default:
        da1501a_parser_reset(parser);
        break;
    }

    return done;
}

static bool range_is_usable(ap_uint<32> distance_mm, bool raw_valid, ap_uint<8> age_frames) {
#pragma HLS INLINE
    bool stale = age_frames > LASER_STALE_FRAME_LIMIT;
    bool blind = distance_mm < LASER_BLIND_ZONE_MM;
    bool out_of_range = (distance_mm < LASER_VALID_DISTANCE_MIN_MM) ||
                        (distance_mm > LASER_VALID_DISTANCE_MAX_MM);
    return raw_valid && !stale && !blind && !out_of_range;
}

static ap_uint<8> make_status_flags(ap_uint<32> distance_mm, bool raw_valid, bool updated, ap_uint<8> age_frames) {
#pragma HLS INLINE
    bool stale = age_frames > LASER_STALE_FRAME_LIMIT;
    bool blind = distance_mm < LASER_BLIND_ZONE_MM;
    bool out_of_range = (distance_mm < LASER_VALID_DISTANCE_MIN_MM) ||
                        (distance_mm > LASER_VALID_DISTANCE_MAX_MM);
    bool usable = raw_valid && !stale && !blind && !out_of_range;

    ap_uint<8> flags = 0;
    if (usable) {
        flags |= (ap_uint<8>)LASER_STATUS_VALID;
    }
    if (blind) {
        flags |= (ap_uint<8>)LASER_STATUS_BLIND_ZONE;
    }
    if (stale) {
        flags |= (ap_uint<8>)LASER_STATUS_STALE;
    }
    if (!usable) {
        flags |= (ap_uint<8>)LASER_STATUS_FALLBACK;
    }
    if (updated) {
        flags |= (ap_uint<8>)LASER_STATUS_UPDATED;
    }
    if (out_of_range) {
        flags |= (ap_uint<8>)LASER_STATUS_OUT_OF_RANGE;
    }
    return flags;
}

static void da1501a_range_update_core(
    ap_uint<8> byte_in,
    bool byte_valid,
    bool frame_tick,
    ap_uint<32> *distance_mm_out,
    ap_uint<8> *status_flags_out,
    ap_uint<8> *range_age_frames_out
) {
#pragma HLS INLINE off
    static da1501a_parser_t parser;
    static bool initialized = false;
    static ap_uint<32> last_distance_mm = 0;
    static ap_uint<8> age_frames = 255;

    if (!initialized) {
        da1501a_parser_reset(&parser);
        initialized = true;
    }

    ap_uint<32> parsed_distance_mm = 0;
    bool updated = false;
    if (byte_valid) {
        updated = da1501a_protocol1_parse_byte(byte_in, &parser, &parsed_distance_mm);
    }

    if (updated) {
        last_distance_mm = parsed_distance_mm;
        age_frames = 0;
    } else if (frame_tick && age_frames < 255) {
        age_frames++;
    }

    bool raw_valid = last_distance_mm != 0;
    *distance_mm_out = last_distance_mm;
    *range_age_frames_out = age_frames;
    *status_flags_out = make_status_flags(last_distance_mm, raw_valid, updated, age_frames);
}

static void set_phase24_fallback_param(laser_param_t *param) {
#pragma HLS INLINE
    param->min_distance_mm = LASER_VALID_DISTANCE_MIN_MM;
    param->max_distance_mm = LASER_VALID_DISTANCE_MAX_MM;
    param->affine_q[0][0] = 67351;
    param->affine_q[0][1] = -8943;
    param->affine_q[0][2] = 20390884;
    param->affine_q[1][0] = 4514;
    param->affine_q[1][1] = 65637;
    param->affine_q[1][2] = 6022770;
    param->fusion_alpha = 96;
    param->lwir_gain_q = 256;
    param->lwir_offset = 0;
}

static void set_phase25_seed_param(laser_param_t *param, int index) {
#pragma HLS INLINE
    set_phase24_fallback_param(param);

    if (index == 0) {
        param->min_distance_mm = 10000;
        param->max_distance_mm = 20000;
        param->affine_q[0][2] = 20543472;
        param->affine_q[1][2] = 5900523;
        param->fusion_alpha = 104;
    } else if (index == 1) {
        param->min_distance_mm = 20000;
        param->max_distance_mm = 50000;
        param->affine_q[0][2] = 20543472;
        param->affine_q[1][2] = 5900523;
        param->fusion_alpha = 104;
    } else if (index == 2) {
        param->min_distance_mm = 50000;
        param->max_distance_mm = 300000;
        param->fusion_alpha = 96;
    } else {
        param->min_distance_mm = 300000;
        param->max_distance_mm = 1500000;
        param->fusion_alpha = 88;
    }
}

static laser_param_t select_laser_param_lut(ap_uint<32> distance_mm, ap_uint<8> range_status_flags) {
#pragma HLS INLINE off
    laser_param_t selected;
    set_phase24_fallback_param(&selected);

    bool range_valid = (range_status_flags & LASER_STATUS_VALID) != 0;
    if (!range_valid) {
        return selected;
    }

    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        laser_param_t candidate;
        set_phase25_seed_param(&candidate, i);
        bool in_bin = (distance_mm >= (ap_uint<32>)candidate.min_distance_mm) &&
                      (distance_mm <= (ap_uint<32>)candidate.max_distance_mm);
        if (in_bin) {
            selected = candidate;
        }
    }

    return selected;
}

static pix8_t sample_lwir_nearest(
    const pix8_t lwir_in[LASER_LWIR_H][LASER_LWIR_W],
    ap_int<32> x_q,
    ap_int<32> y_q
) {
#pragma HLS INLINE
    ap_int<32> x = x_q >> LASER_AFFINE_FRAC_BITS;
    ap_int<32> y = y_q >> LASER_AFFINE_FRAC_BITS;

    if (x < 0 || x >= LASER_LWIR_W || y < 0 || y >= LASER_LWIR_H) {
        return 0;
    }

    return lwir_in[y][x];
}

static rgb_pixel_t fuse_rgb_lwir(rgb_pixel_t rgb, pix8_t lwir, const laser_param_t *param) {
#pragma HLS INLINE
    ap_uint<8> alpha = param->fusion_alpha;
    ap_uint<9> inv_alpha = 256 - alpha;
    ap_int<32> thermal = (((ap_int<32>)lwir * param->lwir_gain_q) >> 8) + param->lwir_offset;
    pix8_t thermal_u8 = clip_u8(thermal);

    rgb_pixel_t out;
    out.r = clip_u8((((ap_int<32>)rgb.r * inv_alpha) + ((ap_int<32>)thermal_u8 * alpha)) >> 8);
    out.g = clip_u8((((ap_int<32>)rgb.g * inv_alpha) + ((ap_int<32>)thermal_u8 * alpha)) >> 8);
    out.b = clip_u8((((ap_int<32>)rgb.b * inv_alpha) + ((ap_int<32>)thermal_u8 * alpha)) >> 8);
    return out;
}

static rgb_pixel_t unpack_rgbx(pix32_t value) {
#pragma HLS INLINE
    rgb_pixel_t out;
    out.r = value.range(7, 0);
    out.g = value.range(15, 8);
    out.b = value.range(23, 16);
    return out;
}

static pix32_t pack_rgbx(rgb_pixel_t value) {
#pragma HLS INLINE
    pix32_t out = 0;
    out.range(7, 0) = value.r;
    out.range(15, 8) = value.g;
    out.range(23, 16) = value.b;
    return out;
}

static void laser_register_fuse_packed_lut_core(
    const pix32_t rgb_in[LASER_OUT_H][LASER_OUT_W],
    const pix8_t lwir_in[LASER_LWIR_H][LASER_LWIR_W],
    pix32_t fused_out[LASER_OUT_H][LASER_OUT_W],
    ap_uint<32> distance_mm,
    ap_uint<8> range_status_flags
) {
#pragma HLS INLINE off
    laser_param_t param = select_laser_param_lut(distance_mm, range_status_flags);

    for (int y = 0; y < LASER_OUT_H; ++y) {
        for (int x = 0; x < LASER_OUT_W; ++x) {
#pragma HLS PIPELINE II=1
            ap_int<64> src_x_q =
                (ap_int<64>)param.affine_q[0][0] * x +
                (ap_int<64>)param.affine_q[0][1] * y +
                (ap_int<64>)param.affine_q[0][2];
            ap_int<64> src_y_q =
                (ap_int<64>)param.affine_q[1][0] * x +
                (ap_int<64>)param.affine_q[1][1] * y +
                (ap_int<64>)param.affine_q[1][2];

            pix8_t lwir = sample_lwir_nearest(lwir_in, (ap_int<32>)src_x_q, (ap_int<32>)src_y_q);
            rgb_pixel_t rgb = unpack_rgbx(rgb_in[y][x]);
            fused_out[y][x] = pack_rgbx(fuse_rgb_lwir(rgb, lwir, &param));
        }
    }
}

void phase25_laser_register_fuse_ip_top(
    const pix32_t rgb_in[LASER_OUT_H][LASER_OUT_W],
    const pix8_t lwir_in[LASER_LWIR_H][LASER_LWIR_W],
    pix32_t fused_out[LASER_OUT_H][LASER_OUT_W],
    ap_uint<64> uart_rx_word,
    ap_uint<4> uart_rx_count,
    bool frame_tick,
    ap_uint<32> *distance_mm_out,
    ap_uint<8> *status_flags_out,
    ap_uint<8> *range_age_frames_out
) {
#pragma HLS INTERFACE m_axi port=rgb_in offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=lwir_in offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=fused_out offset=slave bundle=gmem2
#pragma HLS INTERFACE s_axilite port=uart_rx_word bundle=control
#pragma HLS INTERFACE s_axilite port=uart_rx_count bundle=control
#pragma HLS INTERFACE s_axilite port=frame_tick bundle=control
#pragma HLS INTERFACE ap_vld port=distance_mm_out
#pragma HLS INTERFACE ap_vld port=status_flags_out
#pragma HLS INTERFACE ap_vld port=range_age_frames_out
#pragma HLS INTERFACE s_axilite port=return bundle=control

    ap_uint<32> distance_mm = 0;
    ap_uint<8> status_flags = 0;
    ap_uint<8> age_frames = 0;
    bool updated = false;

    da1501a_range_update_core(
        0,
        false,
        false,
        &distance_mm,
        &status_flags,
        &age_frames
    );

    for (int i = 0; i < LASER_UART_BYTES_PER_WORD; ++i) {
#pragma HLS UNROLL
        if (i < uart_rx_count) {
            ap_uint<8> byte_in = uart_rx_word.range((i * 8) + 7, i * 8);
            da1501a_range_update_core(
                byte_in,
                true,
                false,
                &distance_mm,
                &status_flags,
                &age_frames
            );
            if ((status_flags & (ap_uint<8>)LASER_STATUS_UPDATED) != 0) {
                updated = true;
            }
        }
    }

    if (frame_tick && !updated) {
        da1501a_range_update_core(
            0,
            false,
            true,
            &distance_mm,
            &status_flags,
            &age_frames
        );
    }

    *distance_mm_out = distance_mm;
    *status_flags_out = status_flags;
    *range_age_frames_out = age_frames;

    laser_register_fuse_packed_lut_core(
        rgb_in,
        lwir_in,
        fused_out,
        distance_mm,
        status_flags
    );
}
