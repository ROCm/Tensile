////////////////////////////////////////////////////////////////////////////////
// sgemm NT 128x128x8 w/ full prefetching
////////////////////////////////////////////////////////////////////////////////

// performance
// t = 1200ms but should be ~50ms

////////////////////////////////////////////////////////////////////////////////
// VGPR Assignments
// v0        workitem_x = l0I
// v1        workitem_y = l1J
// v[8:9]    debug address (when enabled)
// v[10:11]  C write base addr
// v[12:13]  C write current addr
// v[14:15]  Global_read_incr x2 incA incB
// v[16:17]  LDS_write_addr   x2 A B
// v[18:19]  LDS_read_addr    x2 A[0:7] B[0:7]
// v[20:23]  Global_read_addr x4 Ax1 Bx1 each 8-byte
// v[24:31]  G->L load regs   x8 Ax4 Bx4 no red/black
// v[32:39]  A_red            x8
// v[40:47]  B_red            x8
// v[48:55]  A_black          x8
// v[56:63]  B_black          x8
// v[64:127] C_MAC            x64 c[a][b]=c[a*8+b] a is row, b is col


////////////////////////////////////////////////////////////////////////////////
// LDS Assignments
// __local float redA[UNROLL*(MACRO_TILE+PAD)]
// size = 8*(128+PAD) = 1032 rounded up to pow2 = 2048 floats
// A red      0000*4 = 0x0000 = 0000000000000000
// B red      2048*4 = 0x2000 = 0010000000000000
// A black    4096*4 = 0x4000 = 0100000000000000
// B black    6144*4 = 0x6000 = 0110000000000000
// swap LDS write:
// v_xor_b32 v16 0x4000 v16
// v_xor_b32 v17 0x4000 v17
// swap LDS read:
// v_xor_b32 v18 0x4000 v18
// v_xor_b32 v19 0x4000 v19


////////////////////////////////////////////////////////////////////////////////
// Kernel Arguments
// s[0:1] - kernelarg_segment
// s2 - workgroup_x = g0I
// s3 - workgroup_y = g1J
//            0-based  24-based
//  C         0x00= 0  0x18=24    s[4:5]
//  A         0x08= 8  0x20=32    s[6:7]
//  B         0x10=16  0x28=40    s[8:9]
//  alpha     0x18=24  0x30=48    s10
//  beta      0x1c=28  0x34=52    s11
//  offsetC   0x20=32  0x38=56    s12
//  offsetA   0x24=36  0x3c=60    s13
//  offsetB   0x28=40  0x40=64    s14
//  strideC1J 0x2c=44  0x44=68    s15
//  strideAK  0x30=48  0x48=72    s16
//  strideBK  0x34=52  0x4c=76    s17
//  size0I    0x38=56  0x50=80    s18 or debug lo
//  size1J    0x3c=60  0x54=84    s19 or debug hi
//  sizeK     0x40=64  0x58=88    s20

////////////////////////////////////////////////////////////////////////////////
// Zero out registers
.macro ZERO_REGISTERS
  .set c, 64
  v_mov_b32 v[c+ 0], 0
  v_mov_b32 v[c+ 1], 0
  v_mov_b32 v[c+ 2], 0
  v_mov_b32 v[c+ 3], 0
  v_mov_b32 v[c+ 4], 0
  v_mov_b32 v[c+ 5], 0
  v_mov_b32 v[c+ 6], 0
  v_mov_b32 v[c+ 7], 0
  v_mov_b32 v[c+ 8], 0
  v_mov_b32 v[c+ 9], 0
  v_mov_b32 v[c+10], 0
  v_mov_b32 v[c+11], 0
  v_mov_b32 v[c+12], 0
  v_mov_b32 v[c+13], 0
  v_mov_b32 v[c+14], 0
  v_mov_b32 v[c+15], 0
  v_mov_b32 v[c+16], 0
  v_mov_b32 v[c+17], 0
  v_mov_b32 v[c+18], 0
  v_mov_b32 v[c+19], 0
  v_mov_b32 v[c+20], 0
  v_mov_b32 v[c+21], 0
  v_mov_b32 v[c+22], 0
  v_mov_b32 v[c+23], 0
  v_mov_b32 v[c+24], 0
  v_mov_b32 v[c+25], 0
  v_mov_b32 v[c+26], 0
  v_mov_b32 v[c+27], 0
  v_mov_b32 v[c+28], 0
  v_mov_b32 v[c+29], 0
  v_mov_b32 v[c+30], 0
  v_mov_b32 v[c+31], 0
  v_mov_b32 v[c+32], 0
  v_mov_b32 v[c+33], 0
  v_mov_b32 v[c+34], 0
  v_mov_b32 v[c+35], 0
  v_mov_b32 v[c+36], 0
  v_mov_b32 v[c+37], 0
  v_mov_b32 v[c+38], 0
  v_mov_b32 v[c+39], 0
  v_mov_b32 v[c+40], 0
  v_mov_b32 v[c+41], 0
  v_mov_b32 v[c+42], 0
  v_mov_b32 v[c+43], 0
  v_mov_b32 v[c+44], 0
  v_mov_b32 v[c+45], 0
  v_mov_b32 v[c+46], 0
  v_mov_b32 v[c+47], 0
  v_mov_b32 v[c+48], 0
  v_mov_b32 v[c+49], 0
  v_mov_b32 v[c+50], 0
  v_mov_b32 v[c+51], 0
  v_mov_b32 v[c+52], 0
  v_mov_b32 v[c+53], 0
  v_mov_b32 v[c+54], 0
  v_mov_b32 v[c+55], 0
  v_mov_b32 v[c+56], 0
  v_mov_b32 v[c+57], 0
  v_mov_b32 v[c+58], 0
  v_mov_b32 v[c+59], 0
  v_mov_b32 v[c+60], 0
  v_mov_b32 v[c+61], 0
  v_mov_b32 v[c+62], 0
  v_mov_b32 v[c+63], 0
.endm

////////////////////////////////////////////////////////////////////////////////
// GL Load G2R - 4 A's and 4 B's
// vm_cnt   += 8
// lgkm_cnt += 8
.macro READ_GLOBAL
  .set src, 20 // Global_read_addr_x4
  .set inc, 14 // Global_read_incr_x2
  .set a, 24
  .set b, 28
  // issue loads global -> registers
.if 0
  // takes 1100ms by itself
  //flat_load_dwordx4 v[a+0:a+3], v[src+0:src+1] // A[0:3]
  //flat_load_dwordx4 v[b+0:b+3], v[src+2:src+3] // B[0:3]

  // +0
  flat_load_dword v[a+0], v[src+0:src+1] // A[0]
  flat_load_dword v[b+0], v[src+2:src+3] // B[0]
  v_add_u32  v[src+0], vcc, v[src+0], v[inc+0] 
  v_addc_u32 v[src+1], vcc, v[src+1], 0x0, vcc
  v_add_u32  v[src+2], vcc, v[src+2], v[inc+1] 
  v_addc_u32 v[src+3], vcc, v[src+3], 0x0, vcc

  // +1
  flat_load_dword v[a+1], v[src+0:src+1] // A[1]
  flat_load_dword v[b+1], v[src+2:src+3] // B[1]
  v_add_u32  v[src+0], vcc, v[src+0], v[inc+0] 
  v_addc_u32 v[src+1], vcc, v[src+1], 0x0, vcc
  v_add_u32  v[src+2], vcc, v[src+2], v[inc+1] 
  v_addc_u32 v[src+3], vcc, v[src+3], 0x0, vcc

  // +2
  flat_load_dword v[a+2], v[src+0:src+1] // A[2]
  flat_load_dword v[b+2], v[src+2:src+3] // B[2]
  v_add_u32  v[src+0], vcc, v[src+0], v[inc+0] 
  v_addc_u32 v[src+1], vcc, v[src+1], 0x0, vcc
  v_add_u32  v[src+2], vcc, v[src+2], v[inc+1] 
  v_addc_u32 v[src+3], vcc, v[src+3], 0x0, vcc

  // +3
  flat_load_dword v[a+3], v[src+0:src+1] // A[3]
  flat_load_dword v[b+3], v[src+2:src+3] // B[3]
  v_add_u32  v[src+0], vcc, v[src+0], v[inc+0] 
  v_addc_u32 v[src+1], vcc, v[src+1], 0x0, vcc
  v_add_u32  v[src+2], vcc, v[src+2], v[inc+1] 
  v_addc_u32 v[src+3], vcc, v[src+3], 0x0, vcc
.endif
.if 0
  v_mov_b32 v[a], 0
  v_add_f32 v[a], 1.0, v[a]
  v_mov_b32 v[a+1], v[a]
  v_mov_b32 v[a+2], v[a]
  v_mov_b32 v[a+3], v[a]
  v_mov_b32 v[b+0], v[a]
  v_mov_b32 v[b+1], v[a]
  v_mov_b32 v[b+2], v[a]
  v_mov_b32 v[b+3], v[a]
.endif

  // increment global addresses for next GL Load
  //v_add_u32  v[src+0], vcc, v[src+0], v[inc+0] 
  //v_addc_u32 v[src+1], vcc, v[src+1], 0x0, vcc
  //v_add_u32  v[src+2], vcc, v[src+2], v[inc+1] 
  //v_addc_u32 v[src+3], vcc, v[src+3], 0x0, vcc

.endm

////////////////////////////////////////////////////////////////////////////////
// GL Load R2L - 4 A's and 4 B's
// lgkm_cnt += 4

.macro WRITE_LOCAL
  .set dst, 16 // LDS_write_addr_x2
  .set a, 24
  .set b, 28

.if 0
  // this code did work
  ds_write2_b64 v[dst+0], v[a+0:a+1], v[a+2:a+3] offset1:1 //32*4*0
  ds_write2_b64 v[dst+1], v[b+0:b+1], v[b+2:b+3] offset1:1 //32*4*0
.endif
.if 0
  // this code did work
  ds_write_b32 v[dst+0], v[a+0] offset:0 //32*4*0
  ds_write_b32 v[dst+0], v[a+1] offset:4 //32*4*1
  ds_write_b32 v[dst+0], v[a+2] offset:8 //32*4*2
  ds_write_b32 v[dst+0], v[a+3] offset:12 //32*4*3
  ds_write_b32 v[dst+1], v[b+0] offset:0 //32*4*0
  ds_write_b32 v[dst+1], v[b+1] offset:4 //32*4*1
  ds_write_b32 v[dst+1], v[b+2] offset:8 //32*4*2
  ds_write_b32 v[dst+1], v[b+3] offset:12 //32*4*3
.endif
.if 0
  // this code works but is slow
  ds_write2st64_b32 v[dst+0], v[a+0], v[a+1] offset0:256*0/64*1 offset1:256*1/64*1
  ds_write2st64_b32 v[dst+0], v[a+2], v[a+3] offset0:256*2/64*1 offset1:256*3/64*1
  ds_write2st64_b32 v[dst+1], v[b+0], v[b+1] offset0:256*0/64*1 offset1:256*1/64*1
  ds_write2st64_b32 v[dst+1], v[b+2], v[b+3] offset0:256*2/64*1 offset1:256*3/64*1

//flat_store_dword v[8:9], v[b+2]
//s_endpgm

  // this code did work
  //ds_write2_b32 v[dst+0], v[a+0], v[a+1]           offset1:1
  //ds_write2_b32 v[dst+0], v[a+2], v[a+3] offset0:2 offset1:3
  //ds_write2_b32 v[dst+1], v[b+0], v[b+1]           offset1:1
  //ds_write2_b32 v[dst+1], v[b+2], v[b+3] offset0:2 offset1:3
.endif
.if 0
  // this code did work
  ds_write_b64 v[dst+0], v[a+0:a+1] //offset:0 //32*4*0
  ds_write_b64 v[dst+0], v[a+2:a+3] offset:8 //32*4*2
  ds_write_b64 v[dst+1], v[b+0:b+1] //offset:0 //32*4*0
  ds_write_b64 v[dst+1], v[b+2:b+3] offset:8 //32*4*2
.endif
.if 0
  // unrecognized instruction mnemonic
  ds_write_b128 v[dst+0], v[a+0:a+3]
  ds_write_b128 v[dst+1], v[b+0:b+3]
.endif


// debug
.if 0
s_waitcnt vmcnt(0) & lgkmcnt(0)
s_barrier
ds_read_b32 v32, v[dst]
s_waitcnt vmcnt(0) & lgkmcnt(0)
s_barrier
flat_store_dword v[8:9], v32
s_endpgm
.endif

.endm

////////////////////////////////////////////////////////////////////////////////
// LR Load L2R - 8 A's and 8 B's
// lgkm_cnt += 4or8
.macro READ_LOCAL gen
  .set src, 18 // LDS_read_addr_x2
  .set inc, (128+0)  // PAD - doesn't appear to impact performance
  .set a, 32
  .set b, 40
  .if \gen%2==1
    .set a, 48
    .set b, 56
  .endif
  // issue loads local -> registers
  // offset is 16 bits and gets multiplied by 4 bytes

// can combine into read2_b32, but then would need to increment
// todo, would it be faster to inverleave data when writing to lds then read block right here
.if 0
  // this is the original working code
  // could change it to read2, but offsets have less bytes
  // and would need more address arithmetic
  ds_read_b32 v[a+0], v[src+0] offset:\gen*inc*4+16*4*0 // A[0]
  ds_read_b32 v[a+1], v[src+0] offset:\gen*inc*4+16*4*1 // A[1]
  ds_read_b32 v[a+2], v[src+0] offset:\gen*inc*4+16*4*2 // A[2]
  ds_read_b32 v[a+3], v[src+0] offset:\gen*inc*4+16*4*3 // A[3]
  ds_read_b32 v[a+4], v[src+0] offset:\gen*inc*4+16*4*4 // A[4]
  ds_read_b32 v[a+5], v[src+0] offset:\gen*inc*4+16*4*5 // A[5]
  ds_read_b32 v[a+6], v[src+0] offset:\gen*inc*4+16*4*6 // A[6]
  ds_read_b32 v[a+7], v[src+0] offset:\gen*inc*4+16*4*7 // A[7]

  ds_read_b32 v[b+0], v[src+1] offset:\gen*inc*4+16*4*0 // B[0]
  ds_read_b32 v[b+1], v[src+1] offset:\gen*inc*4+16*4*1 // B[1]
  ds_read_b32 v[b+2], v[src+1] offset:\gen*inc*4+16*4*2 // B[2]
  ds_read_b32 v[b+3], v[src+1] offset:\gen*inc*4+16*4*3 // B[3]
  ds_read_b32 v[b+4], v[src+1] offset:\gen*inc*4+16*4*4 // B[4]
  ds_read_b32 v[b+5], v[src+1] offset:\gen*inc*4+16*4*5 // B[5]
  ds_read_b32 v[b+6], v[src+1] offset:\gen*inc*4+16*4*6 // B[6]
  ds_read_b32 v[b+7], v[src+1] offset:\gen*inc*4+16*4*7 // B[7]
.endif
.if 0
  // new version where each thread works on contiguous elements
  // unrecognized instruction mnemonic
  ds_read_b128 v[a+0:a+3], v[src+0] offset:\gen*inc*4+4*0 // A[0:3]
  ds_read_b128 v[a+4:a+7], v[src+0] offset:\gen*inc*4+4*4 // A[4:7]
  ds_read_b128 v[b+0:b+3], v[src+1] offset:\gen*inc*4+4*0 // B[0:3]
  ds_read_b128 v[b+4:b+7], v[src+1] offset:\gen*inc*4+4*4 // B[4:7]
.endif

.if 0
  // new version fake, doesn't increment offset
  ds_read2_b64 v[a+4:a+7], v[src+0] offset0:2 offset1:3 // A[4:7]
  ds_read2_b64 v[a+0:a+3], v[src+0]           offset1:1 // A[0:3]
  ds_read2_b64 v[b+0:b+3], v[src+1]           offset1:1 // B[0:3]
  ds_read2_b64 v[b+4:b+7], v[src+1] offset0:2 offset1:3 // B[4:7]
.endif
.if 0
  // new version where each thread works on contiguous elements
  // this code works
  ds_read_b64 v[a+0:a+1], v[src+0] offset:\gen*inc*4+4*0 // A[0:1]
  ds_read_b64 v[a+2:a+3], v[src+0] offset:\gen*inc*4+4*2 // A[2:3]
  ds_read_b64 v[a+4:a+5], v[src+0] offset:\gen*inc*4+4*4 // A[4:5]
  ds_read_b64 v[a+6:a+7], v[src+0] offset:\gen*inc*4+4*6 // A[6:7]
  ds_read_b64 v[b+0:b+1], v[src+1] offset:\gen*inc*4+4*0 // B[0:1]
  ds_read_b64 v[b+2:b+3], v[src+1] offset:\gen*inc*4+4*2 // B[2:3]
  ds_read_b64 v[b+4:b+5], v[src+1] offset:\gen*inc*4+4*4 // B[4:5]
  ds_read_b64 v[b+6:b+7], v[src+1] offset:\gen*inc*4+4*6 // B[6:7]

//s_waitcnt lgkmcnt(0)
//flat_store_dword v[8:9], v[a+0]
//s_endpgm

.endif
.if 0
  v_mov_b32 v[a], 0
  v_add_f32 v[a], 1.0, v[a]
  v_mov_b32 v[a+1], v[a]
  v_mov_b32 v[a+2], v[a]
  v_mov_b32 v[a+3], v[a]
  v_mov_b32 v[a+4], v[a]
  v_mov_b32 v[a+5], v[a]
  v_mov_b32 v[a+6], v[a]
  v_mov_b32 v[a+7], v[a]
  v_mov_b32 v[b+0], v[a]
  v_mov_b32 v[b+1], v[a]
  v_mov_b32 v[b+2], v[a]
  v_mov_b32 v[b+3], v[a]
  v_mov_b32 v[b+4], v[a]
  v_mov_b32 v[b+5], v[a]
  v_mov_b32 v[b+6], v[a]
  v_mov_b32 v[b+7], v[a]
.endif

.endm

////////////////////////////////////////////////////////////////////////////////
// DO 4X4 Quadrant of MACs
.macro MAC_8X8 gen
  .set a, 32 // A_red
  .set b, 40 // B_red
  .set c, 64 // C_MAC
  .if \gen%2 == 1
    .set a, 48 // A_black
    .set b, 56 // B_black
  .endif

  v_mac_f32 v[c+0+0*8], v[a+0], v[b+0]
  v_mac_f32 v[c+1+0*8], v[a+1], v[b+0] 
  v_mac_f32 v[c+2+0*8], v[a+2], v[b+0] 
  v_mac_f32 v[c+3+0*8], v[a+3], v[b+0] 
  v_mac_f32 v[c+4+0*8], v[a+4], v[b+0] 
  v_mac_f32 v[c+5+0*8], v[a+5], v[b+0] 
  v_mac_f32 v[c+6+0*8], v[a+6], v[b+0] 
  v_mac_f32 v[c+7+0*8], v[a+7], v[b+0] 
  v_mac_f32 v[c+7+1*8], v[a+7], v[b+1] 
  v_mac_f32 v[c+6+1*8], v[a+6], v[b+1] 
  v_mac_f32 v[c+5+1*8], v[a+5], v[b+1] 
  v_mac_f32 v[c+4+1*8], v[a+4], v[b+1] 
  v_mac_f32 v[c+3+1*8], v[a+3], v[b+1] 
  v_mac_f32 v[c+2+1*8], v[a+2], v[b+1] 
  v_mac_f32 v[c+1+1*8], v[a+1], v[b+1] 
  v_mac_f32 v[c+0+1*8], v[a+0], v[b+1] 

  v_mac_f32 v[c+0+2*8], v[a+0], v[b+2]
  v_mac_f32 v[c+1+2*8], v[a+1], v[b+2] 
  v_mac_f32 v[c+2+2*8], v[a+2], v[b+2] 
  v_mac_f32 v[c+3+2*8], v[a+3], v[b+2] 
  v_mac_f32 v[c+4+2*8], v[a+4], v[b+2] 
  v_mac_f32 v[c+5+2*8], v[a+5], v[b+2] 
  v_mac_f32 v[c+6+2*8], v[a+6], v[b+2] 
  v_mac_f32 v[c+7+2*8], v[a+7], v[b+2] 
  v_mac_f32 v[c+7+3*8], v[a+7], v[b+3] 
  v_mac_f32 v[c+6+3*8], v[a+6], v[b+3] 
  v_mac_f32 v[c+5+3*8], v[a+5], v[b+3] 
  v_mac_f32 v[c+4+3*8], v[a+4], v[b+3] 
  v_mac_f32 v[c+3+3*8], v[a+3], v[b+3] 
  v_mac_f32 v[c+2+3*8], v[a+2], v[b+3] 
  v_mac_f32 v[c+1+3*8], v[a+1], v[b+3] 
  v_mac_f32 v[c+0+3*8], v[a+0], v[b+3] 

  v_mac_f32 v[c+0+4*8], v[a+0], v[b+4]
  v_mac_f32 v[c+1+4*8], v[a+1], v[b+4] 
  v_mac_f32 v[c+2+4*8], v[a+2], v[b+4] 
  v_mac_f32 v[c+3+4*8], v[a+3], v[b+4] 
  v_mac_f32 v[c+4+4*8], v[a+4], v[b+4] 
  v_mac_f32 v[c+5+4*8], v[a+5], v[b+4] 
  v_mac_f32 v[c+6+4*8], v[a+6], v[b+4] 
  v_mac_f32 v[c+7+4*8], v[a+7], v[b+4] 
  v_mac_f32 v[c+7+5*8], v[a+7], v[b+5] 
  v_mac_f32 v[c+6+5*8], v[a+6], v[b+5] 
  v_mac_f32 v[c+5+5*8], v[a+5], v[b+5] 
  v_mac_f32 v[c+4+5*8], v[a+4], v[b+5] 
  v_mac_f32 v[c+3+5*8], v[a+3], v[b+5] 
  v_mac_f32 v[c+2+5*8], v[a+2], v[b+5] 
  v_mac_f32 v[c+1+5*8], v[a+1], v[b+5] 
  v_mac_f32 v[c+0+5*8], v[a+0], v[b+5] 

  v_mac_f32 v[c+0+6*8], v[a+0], v[b+6]
  v_mac_f32 v[c+1+6*8], v[a+1], v[b+6] 
  v_mac_f32 v[c+2+6*8], v[a+2], v[b+6] 
  v_mac_f32 v[c+3+6*8], v[a+3], v[b+6] 
  v_mac_f32 v[c+4+6*8], v[a+4], v[b+6] 
  v_mac_f32 v[c+5+6*8], v[a+5], v[b+6] 
  v_mac_f32 v[c+6+6*8], v[a+6], v[b+6] 
  v_mac_f32 v[c+7+6*8], v[a+7], v[b+6] 
  v_mac_f32 v[c+7+7*8], v[a+7], v[b+7] 
  v_mac_f32 v[c+6+7*8], v[a+6], v[b+7] 
  v_mac_f32 v[c+5+7*8], v[a+5], v[b+7] 
  v_mac_f32 v[c+4+7*8], v[a+4], v[b+7] 
  v_mac_f32 v[c+3+7*8], v[a+3], v[b+7] 
  v_mac_f32 v[c+2+7*8], v[a+2], v[b+7] 
  v_mac_f32 v[c+1+7*8], v[a+1], v[b+7] 
  v_mac_f32 v[c+0+7*8], v[a+0], v[b+7] 

.endm


////////////////////////////////////////////////////////////////////////////////
// Final Mul/Add/Write
.macro WRITE_GLOBAL d0 d1

// v[10:11] has base address
// v[12:13] will have target address
// target address = base + d0*128 + d1*128*strideC1J
.if 1
.set idx, 64+\d1*8+\d0
v_mov_b32 v12, s15                    // v12 = strideC1J
v_mov_b32 v13, 0x0                    // v13 = 0
v_mul_u32_u24 v12, \d1, v12           // v12 = strideC1J*d1
v_add_u32 v12, vcc, \d0, v12          // v12 = strideC1J*d1+d0
v_lshlrev_b64 v[12:13], 6, v[12:13]   // v12 = 16*(strideC1J*d1+d0)*4
v_add_u32 v12, vcc, v10, v12          // v12 = base + 16*(strideC1J*d1+d0)
v_addc_u32 v13, vcc, v11, v13, vcc    // v13 = base + 16*(strideC1J*d1+d0)
flat_load_dword v9, v[12:13]          // load C
s_waitcnt vmcnt(0) & lgkmcnt(0)       // wait C
v_mul_f32 v9, s11, v9                 // v9 = C*beta
v_mul_f32 v[idx], s10, v[idx]         // v[i] *= alpha
v_add_f32 v[idx], v9, v[idx]          // v[i] = sum*alpha + C*beta
flat_store_dword v[12:13], v[idx]     // store C
// s_waitcnt vmcnt(0) & lgkmcnt(0)       // wait C
.endif
.endm


////////////////////////////////////////////////////////////////////////////////
//  Kernel Descriptor  /////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
.hsa_code_object_version 2,0
.hsa_code_object_isa 8, 0, 3, "AMD", "AMDGPU"
.text
.p2align 8
.amdgpu_hsa_kernel sgemm_NT
sgemm_NT:
.amd_kernel_code_t
  is_ptr64 = 1
  enable_sgpr_kernarg_segment_ptr   =  1  // we do have kernel arguments
  kernarg_segment_byte_size         = 68  // size (bytes) of kernel arguments
  workitem_vgpr_count               = 128 // v127 is max idx
  wavefront_sgpr_count              = 22  // s21 is max idx
  compute_pgm_rsrc1_vgprs           = 31  // floor((128-1)/4)
  compute_pgm_rsrc1_sgprs           = 2   // floor((22-1)/8)
  compute_pgm_rsrc2_user_sgpr       = 2
  compute_pgm_rsrc2_tidig_comp_cnt  = 1 // 2D
  compute_pgm_rsrc2_tgid_x_en       = 1 // preload workgroup.x into sgpr
  compute_pgm_rsrc2_tgid_y_en       = 1
  compute_pgm_rsrc2_lds_size        = 1
  workgroup_group_segment_byte_size = 32768
  kernarg_segment_alignment = 4
  group_segment_alignment = 4
  private_segment_alignment = 4

  //wavefront_size                    = 64
.end_amd_kernel_code_t

////////////////////////////////////////////////////////////////////////////////
//  Begin Kernel: Prepare for Summation Loop  //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

.set wait_for_g2r, 1
.set wait_for_r2l, 1
.set wait_for_l2r, 1

s_mov_b32 m0, 0xFFFFFFFF              // don't clamp LDS

// load kernargs
s_load_dwordx16 s[4:19], s[0:1], 0x0  // load first 16 registers of kernargs
s_load_dword s20, s[0:1], 0x40        // load 17th register of kernargs
s_waitcnt lgkmcnt(0)                  // wait for all kernargs

////////////////////////////////////////////////////////////////////////////////
// debug address
// v[8:9] = D* + serial*16*4
// serial = l.x + g.x*16 + (l.y+g.y*16)*32
// s[18:19] is D*
.if 1
v_mov_b32 v2, s2                      // v2=g0I
v_mov_b32 v3, s3                      // v3=g1J
v_lshlrev_b32 v2, 4, v2               // v2=g0I*16
v_lshlrev_b32 v3, 4, v3               // v3=g1J*16
v_add_i32 v2, vcc, v2, v0             // v2=g0I*16+l0I
v_add_i32 v3, vcc, v3, v1             // v3=g1J*16+l1J
v_mul_lo_u32 v3, 64, v3               // v3=(g1J*16+l1J)*32 // HARDCODED
v_add_u32 v2, vcc, v3, v2             // v2=(g1J*16+l1J)*32 + g0I*16+l0I
v_lshlrev_b32 v2, 2, v2               // v2 16*4
v_mov_b32 v3, 0
v_add_i32 v8, vcc, s18, v2            // v[8:9]=D* + serial*16*4
v_addc_u32 v9, vcc, s19, v3, vcc
//flat_store_dword v[8:9], v2
//s_endpgm

//v_mov_b32 v8, s18
//v_mov_b32 v9, s19
//v_add_u32 v8, vcc, 192, v8            // v[8:9]=D* + serial*16*4
//v_addc_u32 v9, vcc, 0, v9, vcc
.endif



// global_read
v_lshlrev_b32 v7, 4, v1               // v7=lid.y*16
v_add_i32 v7, vcc, v0, v7             // v7=lid.x+lid.y*16
v_and_b32 v2, 127, v7                 // v2=(lid.x+lid.y*16)%128 = a0I, b1J
v_lshrrev_b32 v3, 7, v7               // v3=(lid.x+lid.y*16)/128 = aK, bK
//flat_store_dword v[8:9], v3
//s_endpgm
// these are correct

// global_readA
v_mul_lo_i32 v7, v3, s16              // v7=aK*strideAK
s_lshl_b32 s21, s2, 7                 // s21 = g0I*128
v_or_b32 v4, s21, v2                  // v4=g0I*128+a0I
v_add_i32 v7, vcc, v4, v7             // v7=(g0I*128+a0I) + aK*strideK = A_my
v_add_i32 v4, vcc, s13, v7            // v4=A_my + offsetA
//flat_store_dword v[8:9], v7           // DEBUG
//s_endpgm                              // DEBUG
v_mov_b32 v7, 0                       // v7=0
v_addc_u32 v5, vcc, 0, v7, vcc        // v5=A_my + offsetA hi
v_lshlrev_b64 v[5:6], 2, v[4:5]       // v[5:6]=(A_my+offsetA)*4
v_add_i32 v20, vcc, s6, v5            // v20=A* + (A_my+offsetA)*4 lo
v_mov_b32 v7, s7                      // v7=A* hi
v_addc_u32 v21, vcc, v6, v7, vcc      // v21=A* + (A_my+offsetA)*4 hi
// v[20:21] is global_readA_0:3
// reading 1's from global correctly

// global_readB ?
v_mul_lo_i32 v7, v3, s17              // v7=bK*strideBK
s_lshl_b32 s21, s3, 7                 // s21=g1J*128
v_or_b32 v6, s21, v2                  // v6=g1J*128+b1J (or = plus)
v_add_i32 v7, vcc, v6, v7             // v7=bK*strideBK + (g1J*128+b1J) = B_my
v_add_i32 v22, vcc, s14, v7           // v22=offsetB+B_my lo
//flat_store_dword v[8:9], v22          // DEBUG
//s_endpgm                              // DEBUG
v_mov_b32 v7, 0                       // v7=0
v_addc_u32 v23, vcc, 0, v7, vcc       // v23=offsetB+B_my hi
v_lshlrev_b64 v[22:23], 2, v[22:23]   // v[22:23]=(B_my+offsetB)*4
v_add_i32 v22, vcc, s8, v22           // v22=B* + (B_my+offsetB)*4 lo
v_mov_b32 v6, s9                      // v6=B* hi
v_addc_u32 v23, vcc, v23, v6, vcc     // v23=B* + (B_my+offsetB)*4 hi
// v[22:23] is global_readB_0:3
// NOT reading 1's from global correctly

// global_read_incr ?
v_mov_b32 v14, s16                    // strideAK
v_mov_b32 v15, s17                    // strideBK
v_lshlrev_b32 v14, 3, v14             // v14=strideAK*2*4
v_lshlrev_b32 v15, 3, v15             // v15=strideBK*2*4
.if 0
v_mov_b32 v14, 4*128
v_mov_b32 v15, 4*128
.endif
//flat_store_dword v[8:9], v14
//s_endpgm

// local_writeA,B
// v16 = ((a0I%16)*8 + a0I/16 + aK*129)*4
v_and_b32 v16, 15, v2                 // v16 = a0I%16
v_lshlrev_b32 v16, 3, v16             // v16 = (a0I%16)*8
v_lshrrev_b32 v6, 4, v2               // v6  = a0I/16
v_add_i32 v16, vcc, v6, v16           // v16 = (a0I%16)*8 + a0I/16
v_mov_b32 v6, 0x80                    // v6  = (128+0) // PAD
v_mad_u32_u24 v16, v3, v6, v16        // v16 = (a0I%16)*8 + a0I/16 + 129*aK
v_lshlrev_b32 v16, 2, v16             // v16 = ((a0I%16)*8 + a0I/16 + 129*aK)*4
v_add_u32 v17, vcc, 0x2000, v16       // v17=v16+2048*4=local_writeB_red
//flat_store_dword v[8:9], v16
//s_endpgm



//v_mov_b32 v6, 0x84                    // v6=(128+1)=0x81 (128+4)=0x PAD
//v_mad_u32_u24 v16, v6, v3, v16        // v16=129*aK+a0I
//v_lshlrev_b32 v16, 2, v16             // v16=4*(129*aK+a0I)=local_writeA_red
//v_add_u32 v17, vcc, 0x2000, v16       // v17=v16+2048*4=local_writeB_red
// v[16:17] is local_writeA,B
//flat_store_dword v[8:9], v16
//s_endpgm


// local_readA,B
v_lshlrev_b32 v18, 5, v0              // v18=l.x*8*4=local_readA
v_lshlrev_b32 v19, 5, v1              // v19=l.y*8*4=local_readB
v_add_i32 v19, vcc, 0x2000, v19       // v19=l.y*4+2048*4 bytes
// v[18:19] is local_readA,B
//flat_store_dword v[8:9], v18
//s_endpgm


// iter count
s_lshr_b32 s20, s20, 3                // s20=sizeK/8
s_sub_i32 s20, 0x0, s20               // s20 = -sizeK/8
s_add_u32 s20, s20, 1                 // extra iter w/o prefetch


ZERO_REGISTERS
                                      //                                 lgkm vm
                                      //                                    0  0
// prefetch before loop
READ_GLOBAL                           // Load (a)                           2  2
s_waitcnt vmcnt(0) & lgkmcnt(0)       // Wait (a)                           0  0
WRITE_LOCAL                           // Stor [b]                           2  0
//s_waitcnt vmcnt(0) & lgkmcnt(0)       // Wait (a)                           0  0
//READ_LOCAL 0                          // Load {d}                       06|10  2
//s_waitcnt vmcnt(0) & lgkmcnt(0)       // Wait (a)                           0  0
v_xor_b32 v16, 0x4000, v16            // Swap     local_write_A
v_xor_b32 v17, 0x4000, v17            // Swap     local_write_B

////////////////////////////////////////////////////////////////////////////////
//  Summation Loop  ////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
label_0000:                           // LOOP
                                      //                                    2  0
// Prefetch Next Unroll
READ_GLOBAL                           // Load (c)                           4  2

// Wait For MacroTile To Load
s_waitcnt lgkmcnt(8)                  // Wait [b,l] write local             2  2
s_barrier                             // barrier after write local

READ_LOCAL 0                          // Load {d}                       06|10  2

//  Iter 0
s_waitcnt lgkmcnt(8) // wait for float_load from local to return fast
//                     // so lgkmcnt doesn't overflow
READ_LOCAL 1                          // Load {e}                       10|18  2
s_waitcnt lgkmcnt(8)                  // Wait {d}                        6|10  2
MAC_8X8 0                             // MAC      

//  Iter 1
READ_LOCAL 2                          // Load {f}                       10|18  2
s_waitcnt lgkmcnt(8)                  // Wait {e}                        6|10  2
MAC_8X8 1                             // MAC      

//  Iter 2
READ_LOCAL 3                          // Load {g}                       10|18  2
s_waitcnt lgkmcnt(8)                  // Wait {f}                        6|10  2
MAC_8X8 2                             // MAC      

//  Iter 3
READ_LOCAL 4                          // Load {h}                       10|18  2
s_waitcnt lgkmcnt(8)                  // Wait {g}                        6|10  2
MAC_8X8 3                             // MAC      

//  Iter 4
READ_LOCAL 5                          // Load {i}                       10|18  2
s_waitcnt lgkmcnt(8)                  // Wait {h}                        6|10  2
MAC_8X8 4                             // MAC      

//  Iter 5
READ_LOCAL 6                          // Load {j}                       10|18  2
s_waitcnt lgkmcnt(8)                  // Wait {i}                        6|10  2
MAC_8X8 5                             // MAC      

//  Iter 6
READ_LOCAL 7                          // Load {k}                       10|18  2
s_waitcnt lgkmcnt(8)                  // Wait {j}                        6|10  2
MAC_8X8 6                             // MAC      

// wait for global to register load, issue register to local store
s_waitcnt vmcnt(0)                    // Wait (c) not k                  4| 8  0
WRITE_LOCAL                           // Write[l]                        6|10  0
v_xor_b32 v16, 0x4000, v16            // Swap     local_write_A
v_xor_b32 v17, 0x4000, v17            // Swap     local_write_B
v_xor_b32 v18, 0x4000, v18            // Swap     local_read_A
v_xor_b32 v19, 0x4000, v19            // Swap     local_read_B

//  Iter 7
s_waitcnt lgkmcnt(4)                  // Wait {k} not l                     2  0
MAC_8X8 7                             // MAC      Iter=7

// here vm is 0 and lgkm=2->0 b/c of 

s_add_u32       s20, s20, 1           // Incr     Counter
s_cmp_eq_i32    s20, 0                // Comp     Counter==0
s_cbranch_scc1  label_0001            // Goto     Loop start
s_branch        label_0000            // Goto     After loop

////////////////////////////////////////////////////////////////////////////////
//  Last Iter W/O Prefetch
////////////////////////////////////////////////////////////////////////////////
label_0001:
                                      //                                    2  0
// Wait For MacroTile To Load
s_waitcnt lgkmcnt(0)                  //                                    0  0
s_barrier                             // barrier after store to local
READ_LOCAL 0                          // Load Iter=0

//  Iter 0
READ_LOCAL 1                          // Load Iter=1
s_waitcnt lgkmcnt(8)                  // Wait Iter=0
MAC_8X8 0                             // MAC  Iter=0

//  Iter 1
READ_LOCAL 2                          // Load Iter=2
s_waitcnt lgkmcnt(8)                  // Wait Iter=1
MAC_8X8 1                             // MAC  Iter=1

//  Iter 2
READ_LOCAL 3                          // Load Iter=3
s_waitcnt lgkmcnt(8)                  // Wait Iter=2
MAC_8X8 2                             // MAC  Iter=2

//  Iter 3
READ_LOCAL 4                          // Load Iter=4
s_waitcnt lgkmcnt(8)                  // Wait Iter=3
MAC_8X8 3                             // MAC  Iter=3

//  Iter 4
READ_LOCAL 5                          // Load Iter=5
s_waitcnt lgkmcnt(8)                  // Wait Iter=4
MAC_8X8 4                             // MAC  Iter=4

//  Iter 5
READ_LOCAL 6                          // Load Iter=6
s_waitcnt lgkmcnt(8)                  // Wait Iter=5
MAC_8X8 5                             // MAC  Iter=5

//  Iter 6
READ_LOCAL 7                          // Load Iter=7
s_waitcnt lgkmcnt(8)                  // Wait Iter=6
MAC_8X8 6                             // MAC  Iter=6

//  Iter 7
s_waitcnt lgkmcnt(0)                  // Wait Iter=7
MAC_8X8 7                             // MAC  Iter=7


////////////////////////////////////////////////////////////////////////////////
//  Final C=alpha*A*B+beta*C  //////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

.if 0
s_lshl_b32 s12, s12, 2                // offsetC *= 4bytes
s_add_u32 s4, s12, s4
s_addc_u32 s5, 0x0, s5                // s[4:5] = C* + offset
s_lshl_b32 s16, s2, 9                 // s16 = g0I*128*4
s_lshl_b32 s17, s3, 9                 // s17 = g1J*128*4
s_mul_i32 s17, s15, s17               // s17 = g1J*128*4*strideC1J
s_add_u32 s4, s16, s4                 // s4 = C*+offset+g0I*128
s_addc_u32 s5, 0x0, s5                // s5 = hi
s_add_u32 s4, s17, s4                 // s4 = C*+offset+g0I*128+g1J*128*strC1J
s_addc_u32 s5, 0x0, s5                // s[4:5] = C* + offset + workgroup offset

v_mov_b32 v10, s4                     // v[10:11] = c*+offset+workgroup offset
v_mov_b32 v11, s5                     // hi

s_lshl_b32 s21, s15, 2                // strideC *= 4 bytes
v_mul_lo_u32 v3, s21, v1              // v3  = l1J*strideC1J
v_add_u32 v10, vcc, v3, v10           // v[10:11]=C*+off+wg_off+l1J*stride
v_addc_u32 v11, vcc, 0x0, v11, vcc
v_lshlrev_b32 v0, 2, v0
v_add_u32 v10, vcc, v0, v10           // v[10:11]=C*+off+wg_off+l1j*str+l0I
v_addc_u32 v11, vcc, 0x0, v11, vcc

WRITE_GLOBAL 0, 0
WRITE_GLOBAL 0, 1
WRITE_GLOBAL 0, 2
WRITE_GLOBAL 0, 3
WRITE_GLOBAL 0, 4
WRITE_GLOBAL 0, 5
WRITE_GLOBAL 0, 6
WRITE_GLOBAL 0, 7

WRITE_GLOBAL 1, 0
WRITE_GLOBAL 1, 1
WRITE_GLOBAL 1, 2
WRITE_GLOBAL 1, 3
WRITE_GLOBAL 1, 4
WRITE_GLOBAL 1, 5
WRITE_GLOBAL 1, 6
WRITE_GLOBAL 1, 7

WRITE_GLOBAL 2, 0
WRITE_GLOBAL 2, 1
WRITE_GLOBAL 2, 2
WRITE_GLOBAL 2, 3
WRITE_GLOBAL 2, 4
WRITE_GLOBAL 2, 5
WRITE_GLOBAL 2, 6
WRITE_GLOBAL 2, 7

WRITE_GLOBAL 3, 0
WRITE_GLOBAL 3, 1
WRITE_GLOBAL 3, 2
WRITE_GLOBAL 3, 3
WRITE_GLOBAL 3, 4
WRITE_GLOBAL 3, 5
WRITE_GLOBAL 3, 6
WRITE_GLOBAL 3, 7

WRITE_GLOBAL 4, 0
WRITE_GLOBAL 4, 1
WRITE_GLOBAL 4, 2
WRITE_GLOBAL 4, 3
WRITE_GLOBAL 4, 4
WRITE_GLOBAL 4, 5
WRITE_GLOBAL 4, 6
WRITE_GLOBAL 4, 7

WRITE_GLOBAL 5, 0
WRITE_GLOBAL 5, 1
WRITE_GLOBAL 5, 2
WRITE_GLOBAL 5, 3
WRITE_GLOBAL 5, 4
WRITE_GLOBAL 5, 5
WRITE_GLOBAL 5, 6
WRITE_GLOBAL 5, 7

WRITE_GLOBAL 6, 0
WRITE_GLOBAL 6, 1
WRITE_GLOBAL 6, 2
WRITE_GLOBAL 6, 3
WRITE_GLOBAL 6, 4
WRITE_GLOBAL 6, 5
WRITE_GLOBAL 6, 6
WRITE_GLOBAL 6, 7

WRITE_GLOBAL 7, 0
WRITE_GLOBAL 7, 1
WRITE_GLOBAL 7, 2
WRITE_GLOBAL 7, 3
WRITE_GLOBAL 7, 4
WRITE_GLOBAL 7, 5
WRITE_GLOBAL 7, 6
WRITE_GLOBAL 7, 7
.endif

s_endpgm

////////////////////////////////////////////////////////////////////////////////
//
// sgemm NT 5760x5760x5760
// Fiji Nano 8192 GFlop/s
// none of these load from global memory (huge penalty)
//
// 46.64 ms (101%)
//   - theoretical
//   
// 47.2 (100%) clocks pinned <-- call this peak
//   - plain v_mac_f32 kernel achieves
//        
// 48.0 (98.3%)
//   - gemm kernel
//   - prefix (calculate addresses)
//             
// 48.16 (98%)
//   - prefix = 0.8 ms
//   - write LDS = 0.16 ms
//                  
// 48.36 (97.6%)
//   - prefix
//   - write LDS & wait = 0.2 ms
//                       
// 52.2 (90.4)
//   - prefix
//   - write LDS & wait
//   - read LDS x4 = 3.4 ms OUCH!
//                              
// 51.8 (91.1%)
//   - prefix
//   - write LDS & wait
//   - read LDS x4 & wait = 3.4 ms
//                                     
// 53.0 (89%)
//   - prefix
//   - store 1's rather than global load = 1.2ms FIX READ
//   - write LDS & wait
//   - read LDS x4 & wait
//                                              
// 55.5 (85%)
//   - prefix
//   - store 1's
//   - write LDS & wait
//   - read LDS x4 & wait
//  - write global cumbersome = 2.5ms IMPROVE

