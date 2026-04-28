#include "laser_fusion.hpp"

#include <cstdio>

static rgb_pixel_t rgb_in[LASER_OUT_H][LASER_OUT_W];
static pix32_t rgbx_in[LASER_OUT_H][LASER_OUT_W];
static pix8_t lwir_in[LASER_LWIR_H][LASER_LWIR_W];
static pix32_t fusedx_out[LASER_OUT_H][LASER_OUT_W];

static void set_phase24_fallback(laser_param_t *param) {
    param->min_distance_mm = 10000;
    param->max_distance_mm = 1500000;
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

static void set_phase25_seed(laser_param_t *param) {
    param->min_distance_mm = 10000;
    param->max_distance_mm = 20000;
    param->affine_q[0][0] = 67351;
    param->affine_q[0][1] = -8943;
    param->affine_q[0][2] = 20543472;
    param->affine_q[1][0] = 4514;
    param->affine_q[1][1] = 65637;
    param->affine_q[1][2] = 5900523;
    param->fusion_alpha = 104;
    param->lwir_gain_q = 256;
    param->lwir_offset = 0;
}

static pix8_t expected_lwir_at(const laser_param_t *param, int x, int y) {
    ap_int<64> src_x_q =
        (ap_int<64>)param->affine_q[0][0] * x +
        (ap_int<64>)param->affine_q[0][1] * y +
        (ap_int<64>)param->affine_q[0][2];
    ap_int<64> src_y_q =
        (ap_int<64>)param->affine_q[1][0] * x +
        (ap_int<64>)param->affine_q[1][1] * y +
        (ap_int<64>)param->affine_q[1][2];
    int sx = (int)(src_x_q >> LASER_AFFINE_FRAC_BITS);
    int sy = (int)(src_y_q >> LASER_AFFINE_FRAC_BITS);

    if (sx < 0 || sx >= LASER_LWIR_W || sy < 0 || sy >= LASER_LWIR_H) {
        return 0;
    }
    return lwir_in[sy][sx];
}

static rgb_pixel_t expected_fusion(rgb_pixel_t rgb, pix8_t lwir, const laser_param_t *param) {
    int alpha = (int)param->fusion_alpha;
    int inv_alpha = 256 - alpha;
    int thermal = (((int)lwir * (int)param->lwir_gain_q) >> 8) + (int)param->lwir_offset;
    if (thermal < 0) {
        thermal = 0;
    }
    if (thermal > 255) {
        thermal = 255;
    }

    rgb_pixel_t out;
    out.r = (pix8_t)((((int)rgb.r * inv_alpha) + (thermal * alpha)) >> 8);
    out.g = (pix8_t)((((int)rgb.g * inv_alpha) + (thermal * alpha)) >> 8);
    out.b = (pix8_t)((((int)rgb.b * inv_alpha) + (thermal * alpha)) >> 8);
    return out;
}

static bool same_rgb(rgb_pixel_t a, rgb_pixel_t b) {
    return a.r == b.r && a.g == b.g && a.b == b.b;
}

static pix32_t pack_rgbx_test(rgb_pixel_t value) {
    pix32_t out = 0;
    out.range(7, 0) = value.r;
    out.range(15, 8) = value.g;
    out.range(23, 16) = value.b;
    return out;
}

static rgb_pixel_t unpack_rgbx_test(pix32_t value) {
    rgb_pixel_t out;
    out.r = value.range(7, 0);
    out.g = value.range(15, 8);
    out.b = value.range(23, 16);
    return out;
}

static ap_uint<64> pack_uart_word_test(const ap_uint<8> bytes[LASER_UART_BYTES_PER_WORD]) {
    ap_uint<64> out = 0;
    for (int i = 0; i < LASER_UART_BYTES_PER_WORD; ++i) {
        out.range((i * 8) + 7, i * 8) = bytes[i];
    }
    return out;
}

static int check_parser(void) {
    da1501a_parser_t parser;
    da1501a_parser_reset(&parser);

    const ap_uint<8> good_frame[8] = {
        0x55, 0xAA, 0x88, 0x01, 0xFF, 0x00, 0x7B, 0x02
    };

    ap_uint<32> distance_mm = 0;
    bool got = false;
    for (int i = 0; i < 8; ++i) {
        got = da1501a_protocol1_parse_byte(good_frame[i], &parser, &distance_mm);
    }

    if (!got || distance_mm != 12300) {
        std::printf("Parser good frame failed: got=%d distance=%u\n",
                    got ? 1 : 0, distance_mm.to_uint());
        return 1;
    }

    da1501a_parser_reset(&parser);
    const ap_uint<8> bad_frame[8] = {
        0x55, 0xAA, 0x88, 0x01, 0xFF, 0x00, 0x7B, 0x00
    };
    for (int i = 0; i < 8; ++i) {
        got = da1501a_protocol1_parse_byte(bad_frame[i], &parser, &distance_mm);
    }

    if (got) {
        std::printf("Parser accepted a bad checksum frame\n");
        return 1;
    }

    return 0;
}

static void fill_images(void) {
    for (int y = 0; y < LASER_OUT_H; ++y) {
        for (int x = 0; x < LASER_OUT_W; ++x) {
            rgb_in[y][x].r = (pix8_t)(x & 0xFF);
            rgb_in[y][x].g = (pix8_t)(y & 0xFF);
            rgb_in[y][x].b = (pix8_t)((x + y) & 0xFF);
            rgbx_in[y][x] = pack_rgbx_test(rgb_in[y][x]);
            fusedx_out[y][x] = 0;
        }
    }

    for (int y = 0; y < LASER_LWIR_H; ++y) {
        for (int x = 0; x < LASER_LWIR_W; ++x) {
            lwir_in[y][x] = (pix8_t)((x * 3 + y * 5) & 0xFF);
        }
    }
}

static int check_packed_pixel(const char *name, int x, int y, const laser_param_t *param) {
    pix8_t lwir = expected_lwir_at(param, x, y);
    rgb_pixel_t expected = expected_fusion(rgb_in[y][x], lwir, param);
    rgb_pixel_t actual = unpack_rgbx_test(fusedx_out[y][x]);
    if (!same_rgb(actual, expected)) {
        std::printf("%s mismatch at (%d,%d): actual=(%u,%u,%u) expected=(%u,%u,%u)\n",
                    name,
                    x,
                    y,
                    actual.r.to_uint(),
                    actual.g.to_uint(),
                    actual.b.to_uint(),
                    expected.r.to_uint(),
                    expected.g.to_uint(),
                    expected.b.to_uint());
        return 1;
    }
    return 0;
}

static int check_unified_ip_top(const laser_param_t *phase25_seed, const laser_param_t *fallback) {
    const ap_uint<8> good_frame[LASER_UART_BYTES_PER_WORD] = {
        0x55, 0xAA, 0x88, 0x01, 0xFF, 0x00, 0x7B, 0x02
    };

    ap_uint<32> distance_mm = 0;
    ap_uint<8> flags = 0;
    ap_uint<8> age_frames = 0;

    phase25_laser_register_fuse_ip_top(
        rgbx_in,
        lwir_in,
        fusedx_out,
        pack_uart_word_test(good_frame),
        LASER_UART_BYTES_PER_WORD,
        false,
        &distance_mm,
        &flags,
        &age_frames
    );

    if (distance_mm != 12300 ||
        ((flags & LASER_STATUS_VALID) == 0) ||
        ((flags & LASER_STATUS_UPDATED) == 0) ||
        ((flags & LASER_STATUS_FALLBACK) != 0) ||
        age_frames != 0) {
        std::printf("Unified IP valid range failed: distance=%u flags=0x%02x age=%u\n",
                    distance_mm.to_uint(),
                    flags.to_uint(),
                    age_frames.to_uint());
        return 1;
    }

    if (check_packed_pixel("unified_phase25", 0, 0, phase25_seed) != 0) {
        return 1;
    }
    if (check_packed_pixel("unified_phase25", 320, 240, phase25_seed) != 0) {
        return 1;
    }

    pix32_t valid_sample = fusedx_out[0][0];

    for (int i = 0; i < LASER_STALE_FRAME_LIMIT + 1; ++i) {
        phase25_laser_register_fuse_ip_top(
            rgbx_in,
            lwir_in,
            fusedx_out,
            0,
            0,
            true,
            &distance_mm,
            &flags,
            &age_frames
        );
    }

    if (((flags & LASER_STATUS_VALID) != 0) ||
        ((flags & LASER_STATUS_STALE) == 0) ||
        ((flags & LASER_STATUS_FALLBACK) == 0) ||
        age_frames <= LASER_STALE_FRAME_LIMIT) {
        std::printf("Unified IP stale fallback failed: distance=%u flags=0x%02x age=%u\n",
                    distance_mm.to_uint(),
                    flags.to_uint(),
                    age_frames.to_uint());
        return 1;
    }

    if (check_packed_pixel("unified_fallback", 0, 0, fallback) != 0) {
        return 1;
    }

    if (valid_sample == fusedx_out[0][0]) {
        std::printf("Expected unified valid output to differ from stale fallback at (0,0)\n");
        return 1;
    }

    return 0;
}

int main(void) {
    if (check_parser() != 0) {
        return 1;
    }

    fill_images();

    laser_param_t fallback;
    set_phase24_fallback(&fallback);

    laser_param_t phase25_seed;
    set_phase25_seed(&phase25_seed);

    if (check_unified_ip_top(&phase25_seed, &fallback) != 0) {
        return 1;
    }

    std::printf("tb_laser_fusion PASS\n");
    return 0;
}
