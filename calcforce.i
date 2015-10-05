%module calcforce

%{
    #define SWIG_FILE_WITH_INIT
    #include "calcforce.h"
    #include "stdio.h"
    #include "math.h"
%}

%include "numpy.i"


%init %{
    import_array();
%}
%apply (double* INPLACE_ARRAY1,int DIM1) {
    (double* sum_xforces,int lx),
    (double* sum_yforces,int ly),
    (double* sum_zforces,int lz),
    (double* bond_lengths,int bl_i),
    (double* bond_stretches,int bs_i),
    (double* bond_energies,int be_i),
    (double* node_dilations, int nd_i),
    (double* bond_health,int bhi),
    (double* bond_stiffness,int bsi),
    (double* iso_ext_stiffness,int ies_i),
    (double* pair_health,int phi),
    (double* pair_stiffness,int psi),
    (double* iso_pair_stiffness,int ips),
    (double* iso_pair_coefficients,int ipc),
    (double* pair_p_lengths,int ppli),
    (double* pair_q_lengths,int pqli)}
%apply (double* INPLACE_ARRAY2,int DIM1,int DIM2) {
    (double* Hpq,int hi,int hj),
    (double* bond_vectors,int bv_i,int bv_j),
    (double* pair_p_vectors,int ppvi,int ppvj),
    (double* pair_q_vectors,int pqvi,int pqvj),
    (double* iso_bending,int num_pivots, int ibjj)}
%apply (double* IN_ARRAY2,int DIM1,int DIM2) {
    (double* vpt_offsets,int voi,int voj),
    (double* locations,int loci,int locj)}
%apply (double* IN_ARRAY1,int DIM1) {
    (double* bond_ref_length,int brl_i),
    (double* iso_ext_weights,int iew_i),
    (double* pair_critical_energy,int pcei),
    (double* vpt_wgt_a,int vwa),
    (double* vpt_wgt_b,int vwb),
    (double* vpt_wgt_c,int vwc)}
%apply (int* IN_ARRAY1,int DIM1) {
    (int* vpt_src_a,int vsal),
    (int* vpt_src_b,int vsbl),
    (int* vpt_src_c,int vscl),
    (int* pair_x,int num_pairs),
    (int* pair_p,int npp),
    (int* pair_q,int npq),
    (int* bond1,int nb1),
    (int* bond2,int nb2),
    (int* bond_x,int num_bonds),
    (int* bond_p,int nbp)}
%include "calcforce.h"
