////////////////////////////////////////////////////////////////////////////////
// sgemm NT 128x128x8 w/ full prefetching
// read global: flat_load_dwordx4 x2
// write local: ds_write2_b64     x2
// read local : ds_read2_b64      x4
// perf: 80.4%
////////////////////////////////////////////////////////////////////////////////

// uncomment ds_read on line 384 to drop performance from 97.1% to 94.1%
// any 11 loads reach 97%, any 12 loads reach 94%

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
// iteration 64 v_mac's
.macro ITERATION iter extra_waitcnt
.set read_a, 32
.set read_b, 40
.set mac_a, 48
.set mac_b, 56
.if \iter%2==1
  .set read_a, 48
  .set read_b, 56
  .set mac_a, 32
  .set mac_b, 40
.endif
  
v_mac_f32 v[c+0+0*8], v[mac_a+0], v[mac_b+0]
v_mac_f32 v[c+1+0*8], v[mac_a+1], v[mac_b+0] 
v_mac_f32 v[c+1+1*8], v[mac_a+1], v[mac_b+1] 
v_mac_f32 v[c+0+1*8], v[mac_a+0], v[mac_b+1] 
v_mac_f32 v[c+2+0*8], v[mac_a+2], v[mac_b+0] 
v_mac_f32 v[c+3+0*8], v[mac_a+3], v[mac_b+0] 
v_mac_f32 v[c+3+1*8], v[mac_a+3], v[mac_b+1] 
v_mac_f32 v[c+2+1*8], v[mac_a+2], v[mac_b+1] 
v_mac_f32 v[c+0+2*8], v[mac_a+0], v[mac_b+2]
v_mac_f32 v[c+1+2*8], v[mac_a+1], v[mac_b+2] 
v_mac_f32 v[c+2+2*8], v[mac_a+2], v[mac_b+2] 
v_mac_f32 v[c+3+2*8], v[mac_a+3], v[mac_b+2] 
v_mac_f32 v[c+3+3*8], v[mac_a+3], v[mac_b+3] 
v_mac_f32 v[c+2+3*8], v[mac_a+2], v[mac_b+3] 
v_mac_f32 v[c+1+3*8], v[mac_a+1], v[mac_b+3] 
v_mac_f32 v[c+0+3*8], v[mac_a+0], v[mac_b+3] 

v_mac_f32 v[c+4+0*8], v[mac_a+4], v[mac_b+0] 
v_mac_f32 v[c+5+0*8], v[mac_a+5], v[mac_b+0] 
v_mac_f32 v[c+5+1*8], v[mac_a+5], v[mac_b+1] 
v_mac_f32 v[c+4+1*8], v[mac_a+4], v[mac_b+1] 
v_mac_f32 v[c+4+2*8], v[mac_a+4], v[mac_b+2] 
v_mac_f32 v[c+5+2*8], v[mac_a+5], v[mac_b+2] 
v_mac_f32 v[c+5+3*8], v[mac_a+5], v[mac_b+3] 
v_mac_f32 v[c+4+3*8], v[mac_a+4], v[mac_b+3] 
v_mac_f32 v[c+0+4*8], v[mac_a+0], v[mac_b+4]
v_mac_f32 v[c+1+4*8], v[mac_a+1], v[mac_b+4] 
v_mac_f32 v[c+2+4*8], v[mac_a+2], v[mac_b+4] 
v_mac_f32 v[c+3+4*8], v[mac_a+3], v[mac_b+4] 
v_mac_f32 v[c+4+4*8], v[mac_a+4], v[mac_b+4] 
v_mac_f32 v[c+5+4*8], v[mac_a+5], v[mac_b+4] 
v_mac_f32 v[c+6+4*8], v[mac_a+6], v[mac_b+4] 
v_mac_f32 v[c+7+4*8], v[mac_a+7], v[mac_b+4] 

v_mac_f32 v[c+7+5*8], v[mac_a+7], v[mac_b+5] 
v_mac_f32 v[c+6+5*8], v[mac_a+6], v[mac_b+5] 
v_mac_f32 v[c+5+5*8], v[mac_a+5], v[mac_b+5] 
v_mac_f32 v[c+4+5*8], v[mac_a+4], v[mac_b+5] 
v_mac_f32 v[c+3+5*8], v[mac_a+3], v[mac_b+5] 
v_mac_f32 v[c+2+5*8], v[mac_a+2], v[mac_b+5] 
v_mac_f32 v[c+1+5*8], v[mac_a+1], v[mac_b+5] 
v_mac_f32 v[c+0+5*8], v[mac_a+0], v[mac_b+5] 
v_mac_f32 v[c+6+0*8], v[mac_a+6], v[mac_b+0] 
v_mac_f32 v[c+7+0*8], v[mac_a+7], v[mac_b+0] 
v_mac_f32 v[c+7+1*8], v[mac_a+7], v[mac_b+1] 
v_mac_f32 v[c+6+1*8], v[mac_a+6], v[mac_b+1] 
v_mac_f32 v[c+6+2*8], v[mac_a+6], v[mac_b+2] 
v_mac_f32 v[c+7+2*8], v[mac_a+7], v[mac_b+2] 
v_mac_f32 v[c+7+3*8], v[mac_a+7], v[mac_b+3] 
v_mac_f32 v[c+6+3*8], v[mac_a+6], v[mac_b+3] 

v_mac_f32 v[c+0+6*8], v[mac_a+0], v[mac_b+6]
v_mac_f32 v[c+1+6*8], v[mac_a+1], v[mac_b+6] 
v_mac_f32 v[c+2+6*8], v[mac_a+2], v[mac_b+6] 
v_mac_f32 v[c+3+6*8], v[mac_a+3], v[mac_b+6] 
v_mac_f32 v[c+4+6*8], v[mac_a+4], v[mac_b+6] 
v_mac_f32 v[c+5+6*8], v[mac_a+5], v[mac_b+6] 
v_mac_f32 v[c+6+6*8], v[mac_a+6], v[mac_b+6] 
v_mac_f32 v[c+7+6*8], v[mac_a+7], v[mac_b+6] 
v_mac_f32 v[c+7+7*8], v[mac_a+7], v[mac_b+7] 
v_mac_f32 v[c+6+7*8], v[mac_a+6], v[mac_b+7] 
v_mac_f32 v[c+5+7*8], v[mac_a+5], v[mac_b+7] 
v_mac_f32 v[c+4+7*8], v[mac_a+4], v[mac_b+7] 
v_mac_f32 v[c+3+7*8], v[mac_a+3], v[mac_b+7] 
v_mac_f32 v[c+2+7*8], v[mac_a+2], v[mac_b+7] 
v_mac_f32 v[c+1+7*8], v[mac_a+1], v[mac_b+7] 
v_mac_f32 v[c+0+7*8], v[mac_a+0], v[mac_b+7] 
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
  compute_pgm_rsrc2_user_sgpr       = 2   // ?
  compute_pgm_rsrc2_tidig_comp_cnt  = 1   // 2D
  compute_pgm_rsrc2_tgid_x_en       = 1   // preload workgroup.x into sgpr
  compute_pgm_rsrc2_tgid_y_en       = 1   // preload workgroup.y into sgpr
  compute_pgm_rsrc2_lds_size        = 1   // ?
  workgroup_group_segment_byte_size = 32256 // overriden by runtime
  kernarg_segment_alignment = 4
  group_segment_alignment = 4
  private_segment_alignment = 4
.end_amd_kernel_code_t

////////////////////////////////////////////////////////////////////////////////
//  Begin Kernel: Prepare for Summation Loop  //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
s_mov_b32 m0, 0xFFFFFFFF              // don't clamp LDS
s_load_dwordx16 s[4:19], s[0:1], 0x0  // load first 16 registers of kernargs
s_load_dword s20, s[0:1], 0x40        // load 17th register of kernargs
s_waitcnt lgkmcnt(0)                  // wait for all kernargs

// global_read
v_lshlrev_b32 v7, 4, v1               // v7=lid.y*16
v_add_i32 v7, vcc, v0, v7             // v7=lid.x+lid.y*16
v_and_b32 v2, 31, v7                  // v2=(lid.x+lid.y*16)%32 = a0I, b1J
v_lshrrev_b32 v3, 5, v7               // v3=(lid.x+lid.y*16)/32 = aK, bK

// global_readA
v_mul_lo_i32 v7, v3, s16              // v7=aK*strideAK
s_lshl_b32 s21, s2, 7                 // s21=g0I*128
v_lshlrev_b32 v4, 2, v2               // v4=a0I*4
v_or_b32 v4, s21, v4                  // v4=g0I*128+a0I*4
v_add_i32 v7, vcc, v4, v7             // v7=(g0I*128+a0I*4) + aK*strideK = A_my
v_add_i32 v4, vcc, s13, v7            // v4=A_my + offsetA
v_mov_b32 v7, 0                       // v7=0
v_addc_u32 v5, vcc, 0, v7, vcc        // v5=A_my + offsetA hi
v_lshlrev_b64 v[5:6], 2, v[4:5]       // v[5:6]=(A_my+offsetA)*4
v_add_i32 v20, vcc, s6, v5            // v20=A* + (A_my+offsetA)*4 lo
v_mov_b32 v7, s7                      // v7=A* hi
v_addc_u32 v21, vcc, v6, v7, vcc      // v21=A* + (A_my+offsetA)*4 hi
// v[20:21] is global_readA_0:3

// global_readB ?
v_mul_lo_i32 v7, v3, s17              // v7=bK*strideBK
s_lshl_b32 s21, s3, 7                 // s21=g1J*128
v_lshlrev_b32 v6, 2, v2               // v6=a1J*4
v_or_b32 v6, s21, v6                  // v6=g1J*128+b1J*4 (or = plus)
v_add_i32 v7, vcc, v6, v7             // v7=bK*strideBK + (g1J*128+b1J*4) = B_my
v_add_i32 v22, vcc, s14, v7           // v22=offsetB+B_my lo
v_mov_b32 v7, 0                       // v7=0
v_addc_u32 v23, vcc, 0, v7, vcc       // v23=offsetB+B_my hi
v_lshlrev_b64 v[22:23], 2, v[22:23]   // v[22:23]=(B_my+offsetB)*4
v_add_i32 v22, vcc, s8, v22           // v22=B* + (B_my+offsetB)*4 lo
v_mov_b32 v6, s9                      // v6=B* hi
v_addc_u32 v23, vcc, v23, v6, vcc     // v23=B* + (B_my+offsetB)*4 hi
// v[22:23] is global_readB_0:3

// global_read_incr
v_mov_b32 v14, s16                    // strideAK
v_mov_b32 v15, s17                    // strideBK
v_lshlrev_b32 v14, 5, v14             // v14=strideAK*UNROLL*4
v_lshlrev_b32 v15, 5, v15             // v15=strideBK*UNROLL*4

// local_writeA,B
v_mov_b32 v6, 0x80                    // v6=(128+1)=0x81 (128+4)=0x PAD
v_lshlrev_b32 v16, 2, v2              // v16=a0I*4
v_mad_u32_u24 v16, v6, v3, v16        // v16=129*aK+a0I
v_lshlrev_b32 v16, 2, v16             // v16=4*(129*aK+a0I)=local_writeA_red
v_add_u32 v17, vcc, 0x2000, v16       // v17=v16+2048*4=local_writeB_red
//v[16:17] is local_writeA,B

// local_readA,B
.if 0
v_lshlrev_b32 v7, 4, v1               // v7=lid.y*16
v_add_i32 v7, vcc, v0, v7             // v7=lid.x+lid.y*16
v_lshlrev_b32 v18, 4, v7              // v18=l.x*4*4b=local_readA
v_lshlrev_b32 v19, 4, v7              // v19=l.y*4*4b=local_readB
.else
v_lshlrev_b32 v18, 4, v0              // v18=l.x*4*4b=local_readA
v_lshlrev_b32 v19, 4, v1              // v19=l.y*4*4b=local_readB
.endif
v_add_i32 v19, vcc, 0x2000, v19       // v19=l.y*2*4b+2048*4 bytes
// v[18:19] is local_readA,B

// iter count
s_lshr_b32 s20, s20, 3                // s20=sizeK/8
s_sub_i32 s20, 0x0, s20               // s20 = -sizeK/8
s_add_u32 s20, s20, 1                 // extra iter w/o prefetch

// init c registers to zero
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


// prefetch before loop
//READ_GLOBAL                           // Load (a)                           2  2
s_waitcnt vmcnt(0) & lgkmcnt(0)       // Wait (a)                           0  0
//WRITE_LOCAL                           // Stor [b]                           2  0
v_xor_b32 v16, 0x4000, v16            // Swap     local_write_A
v_xor_b32 v17, 0x4000, v17            // Swap     local_write_B
s_waitcnt lgkmcnt(0)                  // Wait [b]                           0  0
s_barrier

// Load A[0:3],B[0:3]
.set read_a, 32
.set read_b, 40
ds_read2_b64 v[read_a+0:read_a+3], v18 offset0:0*128*4/8 offset1:0*128*4/8+8/8 // A[0:3]
ds_read2_b64 v[read_b+0:read_b+3], v19 offset0:0*128*4/8 offset1:0*128*4/8+8/8 // B[0:3]
ds_read2_b64 v[read_a+4:read_a+7], v18 offset0:0*128*4/8+64*4/8 offset1:0*128*4/8+64*4/8+8/8 // A[4:7]
ds_read2_b64 v[read_b+4:read_b+7], v19 offset0:0*128*4/8+64*4/8 offset1:0*128*4/8+64*4/8+8/8 // B[4:7]
s_waitcnt lgkmcnt(0)


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////
////  Summation Loop
////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
label_0000:
s_barrier // barrier here gets 97.5%
.set c, 64

// Iteration 0 /////////////////////////////////////////////////////////////////
.if 1
s_waitcnt lgkmcnt(0)
ds_read2_b64 v[48+0:48+3], v18 offset0:1*128*4/8        offset1:1*128*4/8+8/8        // A[0:3]
ds_read2_b64 v[48+4:48+7], v18 offset0:1*128*4/8+64*4/8 offset1:1*128*4/8+64*4/8+8/8 // A[4:7]
ds_read2_b64 v[56+0:56+3], v19 offset0:1*128*4/8        offset1:1*128*4/8+8/8        // B[0:3]
ds_read2_b64 v[56+4:56+7], v19 offset0:1*128*4/8+64*4/8 offset1:1*128*4/8+64*4/8+8/8 // B[4:7]
.endif
ITERATION 1 0

// Iteration 1 /////////////////////////////////////////////////////////////////
.if 1
s_waitcnt lgkmcnt(0)
ds_read2_b64 v[32+0:32+3], v18 offset0:2*128*4/8        offset1:2*128*4/8+8/8        // A[0:3]
ds_read2_b64 v[32+4:32+7], v18 offset0:2*128*4/8+64*4/8 offset1:2*128*4/8+64*4/8+8/8 // A[4:7]
ds_read2_b64 v[40+0:40+3], v19 offset0:2*128*4/8        offset1:2*128*4/8+8/8        // B[0:3]
ds_read2_b64 v[40+4:40+7], v19 offset0:2*128*4/8+64*4/8 offset1:2*128*4/8+64*4/8+8/8 // B[4:7]
.endif
ITERATION 2 0

// Iteration 2 /////////////////////////////////////////////////////////////////
.if 1
s_waitcnt lgkmcnt(0)
ds_read2_b64 v[48+0:48+3], v18 offset0:3*128*4/8        offset1:3*128*4/8+8/8        // A[0:3]
ds_read2_b64 v[48+4:48+7], v18 offset0:3*128*4/8+64*4/8 offset1:3*128*4/8+64*4/8+8/8 // A[4:7]
ds_read2_b64 v[56+0:56+3], v19 offset0:3*128*4/8        offset1:3*128*4/8+8/8        // B[0:3]
ds_read2_b64 v[56+4:56+7], v19 offset0:3*128*4/8+64*4/8 offset1:3*128*4/8+64*4/8+8/8 // B[4:7]
.endif
ITERATION 3 0

// Iteration 3 /////////////////////////////////////////////////////////////////
.if 0
s_waitcnt lgkmcnt(0)
ds_read2_b64 v[32+0:32+3], v18 offset0:0*128*4/8        offset1:0*128*4/8+8/8        // A[0:3]
ds_read2_b64 v[32+4:32+7], v18 offset0:0*128*4/8+64*4/8 offset1:0*128*4/8+64*4/8+8/8 // A[4:7]
ds_read2_b64 v[40+0:40+3], v19 offset0:0*128*4/8        offset1:0*128*4/8+8/8        // B[0:3]
ds_read2_b64 v[40+4:40+7], v19 offset0:0*128*4/8+64*4/8 offset1:0*128*4/8+64*4/8+8/8 // B[4:7]
.endif
ITERATION 0 0

// Iteration 4 /////////////////////////////////////////////////////////////////
.if 0
s_waitcnt lgkmcnt(0)
ds_read2_b64 v[48+0:48+3], v18 offset0:1*128*4/8        offset1:1*128*4/8+8/8        // A[0:3]
ds_read2_b64 v[48+4:48+7], v18 offset0:1*128*4/8+64*4/8 offset1:1*128*4/8+64*4/8+8/8 // A[4:7]
ds_read2_b64 v[56+0:56+3], v19 offset0:1*128*4/8        offset1:1*128*4/8+8/8        // B[0:3]
ds_read2_b64 v[56+4:56+7], v19 offset0:1*128*4/8+64*4/8 offset1:1*128*4/8+64*4/8+8/8 // B[4:7]
.endif
ITERATION 1 0

// Iteration 5 /////////////////////////////////////////////////////////////////
.if 0
s_waitcnt lgkmcnt(0)
ds_read2_b64 v[32+0:32+3], v18 offset0:2*128*4/8        offset1:2*128*4/8+8/8        // A[0:3]
ds_read2_b64 v[32+4:32+7], v18 offset0:2*128*4/8+64*4/8 offset1:2*128*4/8+64*4/8+8/8 // A[4:7]
ds_read2_b64 v[40+0:40+3], v19 offset0:2*128*4/8        offset1:2*128*4/8+8/8        // B[0:3]
ds_read2_b64 v[40+4:40+7], v19 offset0:2*128*4/8+64*4/8 offset1:2*128*4/8+64*4/8+8/8 // B[4:7]
.endif
ITERATION 2 0

// Iteration 6 /////////////////////////////////////////////////////////////////
.if 0
s_waitcnt lgkmcnt(0)
ds_read2_b64 v[48+0:48+3], v18 offset0:3*128*4/8        offset1:3*128*4/8+8/8        // A[0:3]
ds_read2_b64 v[48+4:48+7], v18 offset0:3*128*4/8+64*4/8 offset1:3*128*4/8+64*4/8+8/8 // A[4:7]
ds_read2_b64 v[56+0:56+3], v19 offset0:3*128*4/8        offset1:3*128*4/8+8/8        // B[0:3]
ds_read2_b64 v[56+4:56+7], v19 offset0:3*128*4/8+64*4/8 offset1:3*128*4/8+64*4/8+8/8 // B[4:7]
.endif
ITERATION 3 0

// Iteration 7 /////////////////////////////////////////////////////////////////
.if 0
s_waitcnt lgkmcnt(0)
ds_read2_b64 v[32+0:32+3], v18 offset0:0*128*4/8        offset1:0*128*4/8+8/8        // A[0:3]
ds_read2_b64 v[32+4:32+7], v18 offset0:0*128*4/8+64*4/8 offset1:0*128*4/8+64*4/8+8/8 // A[4:7]
ds_read2_b64 v[40+0:40+3], v19 offset0:0*128*4/8        offset1:0*128*4/8+8/8        // B[0:3]
ds_read2_b64 v[40+4:40+7], v19 offset0:0*128*4/8+64*4/8 offset1:0*128*4/8+64*4/8+8/8 // B[4:7]
.endif
ITERATION 0 0


// swap lds addresses
v_xor_b32 v16, 0x4000, v16            // Swap     local_write_A
v_xor_b32 v17, 0x4000, v17            // Swap     local_write_B
v_xor_b32 v18, 0x4000, v18            // Swap     local_read_A
v_xor_b32 v19, 0x4000, v19            // Swap     local_read_B

// loop back to beginning
s_add_u32       s20, s20, 1           // Incr     Counter
s_cmp_eq_i32    s20, 0                // Comp     Counter==0
//s_barrier // barrier here gets 97.5%
s_cbranch_scc1  label_0001            // Goto     Loop start
s_branch        label_0000            // Goto     After loop

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////
////  Last Iter W/O Prefetch
////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
label_0001:
                                      //                                    2  0
// Wait For MacroTile To Load
//s_waitcnt lgkmcnt(2)                  //                                    0  0
//s_barrier                             // barrier after store to local

// Iteration 1-7 ///////////////////////////////////////////////////////////////
ITERATION 1 0
ITERATION 2 0
ITERATION 3 0
ITERATION 0 0
ITERATION 1 0
ITERATION 2 0
ITERATION 3 0

// Iteration 8 /////////////////////////////////////////////////////////////////

// Wait A[0:3],B[0:3]. A[4:7],B[4:7] outstanding.
// MACs A[0:3],B[0:3].
s_waitcnt lgkmcnt(8)
.set mac_a, 48
.set mac_b, 56
v_mac_f32 v[c+0+0*8], v[mac_a+0], v[mac_b+0]
v_mac_f32 v[c+1+0*8], v[mac_a+1], v[mac_b+0] 
v_mac_f32 v[c+1+1*8], v[mac_a+1], v[mac_b+1] 
v_mac_f32 v[c+0+1*8], v[mac_a+0], v[mac_b+1] 
v_mac_f32 v[c+2+0*8], v[mac_a+2], v[mac_b+0] 
v_mac_f32 v[c+3+0*8], v[mac_a+3], v[mac_b+0] 
v_mac_f32 v[c+3+1*8], v[mac_a+3], v[mac_b+1] 
v_mac_f32 v[c+2+1*8], v[mac_a+2], v[mac_b+1] 
v_mac_f32 v[c+0+2*8], v[mac_a+0], v[mac_b+2]
v_mac_f32 v[c+1+2*8], v[mac_a+1], v[mac_b+2] 
v_mac_f32 v[c+2+2*8], v[mac_a+2], v[mac_b+2] 
v_mac_f32 v[c+3+2*8], v[mac_a+3], v[mac_b+2] 
v_mac_f32 v[c+3+3*8], v[mac_a+3], v[mac_b+3] 
v_mac_f32 v[c+2+3*8], v[mac_a+2], v[mac_b+3] 
v_mac_f32 v[c+1+3*8], v[mac_a+1], v[mac_b+3] 
v_mac_f32 v[c+0+3*8], v[mac_a+0], v[mac_b+3] 

// Wait A[4:7],B[4:7]. None outstanding.
// MACs A[4:7],B[4:7].
s_waitcnt lgkmcnt(0)
v_mac_f32 v[c+4+0*8], v[mac_a+4], v[mac_b+0] 
v_mac_f32 v[c+5+0*8], v[mac_a+5], v[mac_b+0] 
v_mac_f32 v[c+6+0*8], v[mac_a+6], v[mac_b+0] 
v_mac_f32 v[c+7+0*8], v[mac_a+7], v[mac_b+0] 
v_mac_f32 v[c+7+1*8], v[mac_a+7], v[mac_b+1] 
v_mac_f32 v[c+6+1*8], v[mac_a+6], v[mac_b+1] 
v_mac_f32 v[c+5+1*8], v[mac_a+5], v[mac_b+1] 
v_mac_f32 v[c+4+1*8], v[mac_a+4], v[mac_b+1] 
v_mac_f32 v[c+4+2*8], v[mac_a+4], v[mac_b+2] 
v_mac_f32 v[c+5+2*8], v[mac_a+5], v[mac_b+2] 
v_mac_f32 v[c+6+2*8], v[mac_a+6], v[mac_b+2] 
v_mac_f32 v[c+7+2*8], v[mac_a+7], v[mac_b+2] 
v_mac_f32 v[c+7+3*8], v[mac_a+7], v[mac_b+3] 
v_mac_f32 v[c+6+3*8], v[mac_a+6], v[mac_b+3] 
v_mac_f32 v[c+5+3*8], v[mac_a+5], v[mac_b+3] 
v_mac_f32 v[c+4+3*8], v[mac_a+4], v[mac_b+3] 
v_mac_f32 v[c+0+4*8], v[mac_a+0], v[mac_b+4]
v_mac_f32 v[c+1+4*8], v[mac_a+1], v[mac_b+4] 
v_mac_f32 v[c+2+4*8], v[mac_a+2], v[mac_b+4] 
v_mac_f32 v[c+3+4*8], v[mac_a+3], v[mac_b+4] 
v_mac_f32 v[c+4+4*8], v[mac_a+4], v[mac_b+4] 
v_mac_f32 v[c+5+4*8], v[mac_a+5], v[mac_b+4] 
v_mac_f32 v[c+6+4*8], v[mac_a+6], v[mac_b+4] 
v_mac_f32 v[c+7+4*8], v[mac_a+7], v[mac_b+4] 
v_mac_f32 v[c+7+5*8], v[mac_a+7], v[mac_b+5] 
v_mac_f32 v[c+6+5*8], v[mac_a+6], v[mac_b+5] 
v_mac_f32 v[c+5+5*8], v[mac_a+5], v[mac_b+5] 
v_mac_f32 v[c+4+5*8], v[mac_a+4], v[mac_b+5] 
v_mac_f32 v[c+3+5*8], v[mac_a+3], v[mac_b+5] 
v_mac_f32 v[c+2+5*8], v[mac_a+2], v[mac_b+5] 
v_mac_f32 v[c+1+5*8], v[mac_a+1], v[mac_b+5] 
v_mac_f32 v[c+0+5*8], v[mac_a+0], v[mac_b+5] 
v_mac_f32 v[c+0+6*8], v[mac_a+0], v[mac_b+6]
v_mac_f32 v[c+1+6*8], v[mac_a+1], v[mac_b+6] 
v_mac_f32 v[c+2+6*8], v[mac_a+2], v[mac_b+6] 
v_mac_f32 v[c+3+6*8], v[mac_a+3], v[mac_b+6] 
v_mac_f32 v[c+4+6*8], v[mac_a+4], v[mac_b+6] 
v_mac_f32 v[c+5+6*8], v[mac_a+5], v[mac_b+6] 
v_mac_f32 v[c+6+6*8], v[mac_a+6], v[mac_b+6] 
v_mac_f32 v[c+7+6*8], v[mac_a+7], v[mac_b+6] 
v_mac_f32 v[c+7+7*8], v[mac_a+7], v[mac_b+7] 
v_mac_f32 v[c+6+7*8], v[mac_a+6], v[mac_b+7] 
v_mac_f32 v[c+5+7*8], v[mac_a+5], v[mac_b+7] 
v_mac_f32 v[c+4+7*8], v[mac_a+4], v[mac_b+7] 
v_mac_f32 v[c+3+7*8], v[mac_a+3], v[mac_b+7] 
v_mac_f32 v[c+2+7*8], v[mac_a+2], v[mac_b+7] 
v_mac_f32 v[c+1+7*8], v[mac_a+1], v[mac_b+7] 
v_mac_f32 v[c+0+7*8], v[mac_a+0], v[mac_b+7] 



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////
////  Final C=alpha*A*B+beta*C
////
////////////////////////////////////////////////////////////////////////////////
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
