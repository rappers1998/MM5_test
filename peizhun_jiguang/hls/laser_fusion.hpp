#pragma once

#include <ap_int.h>

#define LASER_OUT_W 640
#define LASER_OUT_H 480
#define LASER_LWIR_W 640
#define LASER_LWIR_H 512
#define LASER_MAX_RANGE_BINS 8
#define LASER_AFFINE_FRAC_BITS 16
#define LASER_VALID_DISTANCE_MIN_MM 10000
#define LASER_VALID_DISTANCE_MAX_MM 1500000
#define LASER_BLIND_ZONE_MM 10000
#define LASER_STALE_FRAME_LIMIT 5
#define LASER_STATUS_VALID 0x01
#define LASER_STATUS_BLIND_ZONE 0x02
#define LASER_STATUS_STALE 0x04
#define LASER_STATUS_FALLBACK 0x08
#define LASER_STATUS_UPDATED 0x10
#define LASER_STATUS_OUT_OF_RANGE 0x20
#define LASER_UART_BYTES_PER_WORD 8

typedef ap_uint<8> pix8_t;
typedef ap_uint<32> pix32_t;

struct rgb_pixel_t {
    pix8_t r;
    pix8_t g;
    pix8_t b;
};

struct laser_param_t {
    ap_int<32> min_distance_mm;
    ap_int<32> max_distance_mm;
    ap_int<32> affine_q[2][3];
    ap_uint<8> fusion_alpha;
    ap_int<16> lwir_gain_q;
    ap_int<16> lwir_offset;
};

struct range_status_t {
    ap_uint<32> distance_mm;
    bool valid;
    bool blind_zone;
    bool stale;
    bool fallback;
};

struct da1501a_parser_t {
    ap_uint<4> state;
    ap_uint<8> cmd;
    ap_uint<8> status;
    ap_uint<8> reserved;
    ap_uint<8> data_h;
    ap_uint<8> data_l;
    ap_uint<16> checksum_acc;
};

void da1501a_parser_reset(da1501a_parser_t *parser);

bool da1501a_protocol1_parse_byte(
    ap_uint<8> byte_in,
    da1501a_parser_t *parser,
    ap_uint<32> *distance_mm
);

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
);
