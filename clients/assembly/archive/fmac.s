// Test Hiding Global Memory Latency



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
  workitem_vgpr_count               = 32  // v127 is max idx
  wavefront_sgpr_count              = 22  // s21 is max idx
  compute_pgm_rsrc1_vgprs           =  7  // floor((80-1)/4)
  compute_pgm_rsrc1_sgprs           =  2  // floor((22-1)/8)
  compute_pgm_rsrc2_user_sgpr       =  2
  compute_pgm_rsrc2_tidig_comp_cnt  =  1  // 2D
  compute_pgm_rsrc2_tgid_x_en       =  1  // preload workgroup.x into sgpr
  compute_pgm_rsrc2_tgid_y_en       =  1
  compute_pgm_rsrc2_lds_size        =  1
  workgroup_group_segment_byte_size = 32768
  kernarg_segment_alignment = 4
  group_segment_alignment = 4
  private_segment_alignment = 4
.end_amd_kernel_code_t



.macro MAC_1
v_mac_f32 v0, v0, v1
.endm
.macro MAC_8
MAC_1
MAC_1
MAC_1
MAC_1
MAC_1
MAC_1
MAC_1
MAC_1
.endm
.macro MAC_64
MAC_8
MAC_8
MAC_8
MAC_8
MAC_8
MAC_8
MAC_8
MAC_8
.endm
.macro MAC_512
MAC_64
MAC_64
MAC_64
MAC_64
MAC_64
MAC_64
MAC_64
MAC_64
.endm

////////////////////////////////////////////////////////////////////////////////
//  Begin Kernel: Prepare for Summation Loop  //////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// load kernargs
s_load_dwordx16 s[4:19], s[0:1], 0x0  // load first 16 registers of kernargs
s_load_dword s20, s[0:1], 0x40        // load 17th register of kernargs
s_waitcnt lgkmcnt(0)                  // wait for all kernargs

.if 1
// sideways read

// global_read
v_lshlrev_b32 v7, 4, v1               // v7=lid.y*16
v_add_i32 v7, vcc, v0, v7             // v7=lid.x+lid.y*16
v_lshrrev_b32 v3, 7, v7               // v3=(lid.x+lid.y*16)/32 = aK, bK
v_and_b32 v2, 127, v7                  // v2=(lid.x+lid.y*16)%32 = a0I, b1J

// global_readA
v_mul_lo_i32 v7, v3, s16              // v7=aK*strideAK
s_lshl_b32 s21, s2, 7                 // s21 = g0I*128
v_lshlrev_b32 v4, 0, v2               // v4=a0I*1
v_or_b32 v4, s21, v4                  // v4=g0I*128+a0I
v_add_i32 v7, vcc, v4, v7             // v7=(g0I*128+a0I) + aK*strideK = A_my
v_add_i32 v4, vcc, s13, v7            // v4=A_my + offsetA
v_mov_b32 v7, 0                       // v7=0
v_addc_u32 v5, vcc, 0, v7, vcc        // v5=A_my + offsetA hi
v_lshlrev_b64 v[5:6], 2, v[4:5]       // v[5:6]=(A_my+offsetA)*4
v_add_i32 v8, vcc, s6, v5            // v8=A* + (A_my+offsetA)*4 lo
v_mov_b32 v7, s7                      // v7=A* hi
v_addc_u32 v9, vcc, v6, v7, vcc      // v21=A* + (A_my+offsetA)*4 hi
// v[8:9] is global_readA_0:3
// reading 1's from global correctly

// add on stride
s_lshl_b32 s4, s16, 3                 // s21=strideK*2*4
//s_mov_b32 s4, 128

v_add_i32 v10, vcc, s4, v8            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v11, vcc, 0, v9, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v12, vcc, s4, v10            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v13, vcc, 0, v11, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v14, vcc, s4, v12            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v15, vcc, 0, v13, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v16, vcc, s4, v14            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v17, vcc, 0, v15, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v18, vcc, s4, v16            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v19, vcc, 0, v17, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v20, vcc, s4, v18            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v21, vcc, 0, v19, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v22, vcc, s4, v20            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v23, vcc, 0, v21, vcc      // v21=A* + (A_my+offsetA)*4 hi


.if 0
// global_readB ?
v_mul_lo_i32 v7, v3, s17              // v7=bK*strideBK
s_lshl_b32 s21, s3, 7                 // s21=g1J*128
v_lshlrev_b32 v6, 0, v2               // v6=a1J*1
v_or_b32 v6, s21, v6                  // v6=g1J*16+b1J*1 (or = plus)
v_add_i32 v7, vcc, v6, v7             // v7=bK*strideBK + (g1J*16+b1J*1) = B_my
v_add_i32 v16, vcc, s14, v7           // v16=offsetB+B_my lo
v_mov_b32 v7, 0                       // v7=0
v_addc_u32 v17, vcc, 0, v7, vcc       // v17=offsetB+B_my hi
v_lshlrev_b64 v[16:17], 2, v[16:17]   // v[16:17]=(B_my+offsetB)*4
v_add_i32 v16, vcc, s8, v16           // v16=B* + (B_my+offsetB)*4 lo
v_mov_b32 v6, s9                      // v6=B* hi
v_addc_u32 v17, vcc, v17, v6, vcc     // v17=B* + (B_my+offsetB)*4 hi
// v[16:17] is global_readB_0:3
// NOT reading 1's from global correctly

v_add_i32 v18, vcc, s4, v16            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v19, vcc, 0, v17, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v20, vcc, s4, v18            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v21, vcc, 0, v19, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v22, vcc, s4, v20            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v23, vcc, 0, v21, vcc      // v21=A* + (A_my+offsetA)*4 hi
.endif

.else

// Cobalt read pattern

// global_read
v_lshlrev_b32 v7, 4, v1               // v7=lid.y*16
v_add_i32 v7, vcc, v0, v7             // v7=lid.x+lid.y*16
v_lshrrev_b32 v3, 5, v7               // v3=(lid.x+lid.y*16)/32 = aK, bK
v_and_b32 v2, 31, v7                  // v2=(lid.x+lid.y*16)%32 = a0I, b1J

// global_readA
v_mul_lo_i32 v7, v3, s16              // v7=aK*strideAK
s_lshl_b32 s21, s2, 7                 // s21 = g0I*128
v_lshlrev_b32 v4, 0, v2               // v4=a0I*1
v_or_b32 v4, s21, v4                  // v4=g0I*16+a0I
v_add_i32 v7, vcc, v4, v7             // v7=(g0I*16+a0I) + aK*strideK = A_my
v_add_i32 v4, vcc, s13, v7            // v4=A_my + offsetA
v_mov_b32 v7, 0                       // v7=0
v_addc_u32 v5, vcc, 0, v7, vcc        // v5=A_my + offsetA hi
v_lshlrev_b64 v[5:6], 2, v[4:5]       // v[5:6]=(A_my+offsetA)*4
v_add_i32 v8, vcc, s6, v5            // v8=A* + (A_my+offsetA)*4 lo
v_mov_b32 v7, s7                      // v7=A* hi
v_addc_u32 v9, vcc, v6, v7, vcc      // v21=A* + (A_my+offsetA)*4 hi
// v[8:9] is global_readA_0:3
// reading 1's from global correctly

v_add_i32 v10, vcc, 32*4, v8            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v11, vcc, 0, v9, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v12, vcc, 32*4, v10            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v13, vcc, 0, v11, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v14, vcc, 32*4, v12            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v15, vcc, 0, v13, vcc      // v21=A* + (A_my+offsetA)*4 hi

// global_readB ?
v_mul_lo_i32 v7, v3, s17              // v7=bK*strideBK
s_lshl_b32 s21, s3, 7                 // s21=g1J*128
v_lshlrev_b32 v6, 0, v2               // v6=a1J*1
v_or_b32 v6, s21, v6                  // v6=g1J*16+b1J*1 (or = plus)
v_add_i32 v7, vcc, v6, v7             // v7=bK*strideBK + (g1J*16+b1J*1) = B_my
v_add_i32 v16, vcc, s14, v7           // v16=offsetB+B_my lo
v_mov_b32 v7, 0                       // v7=0
v_addc_u32 v17, vcc, 0, v7, vcc       // v17=offsetB+B_my hi
v_lshlrev_b64 v[16:17], 2, v[16:17]   // v[16:17]=(B_my+offsetB)*4
v_add_i32 v16, vcc, s8, v16           // v16=B* + (B_my+offsetB)*4 lo
v_mov_b32 v6, s9                      // v6=B* hi
v_addc_u32 v17, vcc, v17, v6, vcc     // v17=B* + (B_my+offsetB)*4 hi
// v[16:17] is global_readB_0:3
// NOT reading 1's from global correctly

v_add_i32 v18, vcc, 32*4, v16            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v19, vcc, 0, v17, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v20, vcc, 32*4, v18            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v21, vcc, 0, v19, vcc      // v21=A* + (A_my+offsetA)*4 hi

v_add_i32 v22, vcc, 32*4, v20            // v8=A* + (A_my+offsetA)*4 lo
v_addc_u32 v23, vcc, 0, v21, vcc      // v21=A* + (A_my+offsetA)*4 hi


.endif

// global_read_incr ?
v_mov_b32 v3, s16                    // strideAK
v_lshlrev_b32 v3, 6, v3             // v3=strideAK*UNROLL=16*4
.if 0
v_mov_b32 v3, 4*128
.endif

s_waitcnt lgkmcnt(0)                  // wait for all kernargs

// iter count
s_lshr_b32 s20, s20, 4                // s20=sizeK/8
s_sub_i32 s20, 0x0, s20               // s20 = -sizeK/8


////////////////////////////////////////////////////////////////////////////////
//  Summation Loop  ////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
label_0000:                           // LOOP

// wait for prior load
s_waitcnt 0

// global memory load
flat_load_dword   v[24], v[8:9]   // A0
flat_load_dword   v[25], v[10:11] // A1
flat_load_dword   v[26], v[12:13] // A2
flat_load_dword   v[27], v[14:15] // A3
flat_load_dword   v[28], v[16:17] // B0
flat_load_dword   v[29], v[18:19] // B1
flat_load_dword   v[30], v[20:21] // B2
flat_load_dword   v[31], v[22:23] // B3


// increment global memory address
v_add_u32   v8, vcc,  v3,  v8
v_addc_u32  v9, vcc,  v9, 0x0, vcc

v_add_u32  v10, vcc,  v3, v10
v_addc_u32 v11, vcc, v11, 0x0, vcc

v_add_u32  v12, vcc,  v3, v12
v_addc_u32 v13, vcc, v13, 0x0, vcc

v_add_u32  v14, vcc, v3, v14
v_addc_u32 v15, vcc, v15, 0x0, vcc

v_add_u32  v16, vcc, v3, v16
v_addc_u32 v17, vcc, v17, 0x0, vcc

v_add_u32  v18, vcc, v3, v18
v_addc_u32 v19, vcc, v19, 0x0, vcc

v_add_u32  v20, vcc, v3, v20
v_addc_u32 v21, vcc, v21, 0x0, vcc

v_add_u32  v22, vcc, v3, v22
v_addc_u32 v23, vcc, v23, 0x0, vcc

// 512 v_mac_f32
//MAC_512
//MAC_512
//MAC_512

//
s_add_u32       s20, s20, 1           // Incr     Counter
s_cmp_eq_i32    s20, 0                // Comp     Counter==0
s_cbranch_scc1  label_0001            // Goto     Loop start
s_branch        label_0000            // Goto     After loop
s_nop 0

label_0001:
// Done, don't write anything to global

s_endpgm
