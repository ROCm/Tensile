////////////////////////////////////////////////////////////////////////////////
// sgemm NT 128x128x8 w/ full prefetching
// read global: flat_load_dwordx4 x2
// write local: ds_write2_b64     x2
// read local : ds_read2_b64      x4
// perf: 80.4%
////////////////////////////////////////////////////////////////////////////////

// TODO didn't correct read global or write local for validation

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

// iter count
s_lshr_b32 s20, s20, 3                // s20=sizeK/8
s_sub_i32 s20, 0x0, s20               // s20 = -sizeK/8



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////
////  Summation Loop
////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
label_0000:
.set c, 64

ITERATION 1 0
ITERATION 2 0
ITERATION 3 0
ITERATION 0 0
ITERATION 1 0
ITERATION 2 0
ITERATION 3 0
ITERATION 0 0

// loop back to beginning
s_add_u32       s20, s20, 1           // Incr     Counter
s_cmp_eq_i32    s20, 0                // Comp     Counter==0
//s_barrier // barrier here gets 97.5%
s_cbranch_scc1  label_0001            // Goto     Loop start
s_branch        label_0000            // Goto     After loop

label_0001:

s_endpgm
