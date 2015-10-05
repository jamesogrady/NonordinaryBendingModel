/*
The MIT License (MIT)

Copyright (c) 2015 James O'Grady

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
void bondvectors(
    double* locations,int loci,int locj,
    double* bond_vectors,int bv_i,int bv_j,
    double* bond_lengths,int bl_i,
    int * bond_x,int num_bonds,
    int * bond_p,int nbp);

void bondstretches(
    double* bond_vectors,int bv_i,int bv_j,
    double* bond_lengths,int bl_i,
    int * bond_x,int num_bonds,
    double* bond_ref_length,int brl_i,
    double* bond_stretches,int bs_i,
    double* iso_ext_weights,int iew_i,
    double* node_dilations, int nd_i);

void bondforces(double* sum_xforces,int lx,
    double* sum_yforces,int ly,
    double* sum_zforces,int lz,
    double* bond_vectors,int bv_i,int bv_j,
    double* bond_lengths,int bl_i,
    double* bond_health,int bhi,
    double* bond_stiffness,int bsi,
    double* bond_stretches,int bs_i,
    double* iso_ext_weights,int iew_i,
    double* iso_ext_stiffness,int ies_i,
    double* node_dilations, int nd_i,
    int * bond_x,int num_bonds,
    int * bond_p,int nbp);

void pairvectors(
    double* locations,int loci,int locj,
    double* vpt_offsets,int voi,int voj,
    double* pair_p_vectors,int ppvi,int ppvj,
    double* pair_q_vectors,int pqvi,int pqvj,
    double* pair_p_lengths,int ppli,
    double* pair_q_lengths,int pqli,
    int * vpt_src_a,int vsal,
    int * vpt_src_b,int vsbl,
    int * vpt_src_c,int vscl,
    double* vpt_wgt_a,int vwa,
    double* vpt_wgt_b,int vwb,
    double* vpt_wgt_c,int vwc,
    int * pair_x,int num_pairs,
    int * pair_p,int npp,
    int * pair_q,int npq);

void pairbending(
    double* Hpq,int hi, int hj,
    double* pair_p_vectors,int ppvi,int ppvj,
    double* pair_q_vectors,int pqvi,int pqvj,
    double* pair_p_lengths,int ppli,
    double* pair_q_lengths,int pqli,
    double* pair_health,int phi, 
    double* pair_stiffness,int psi,
    double* iso_pair_coefficients,int ipc,
    double* iso_bending,int num_pivots, int ibjj,
    int * pair_x,int num_pairs);

void pairbreakbrittle(
    double* Hpq,int hi, int hj,
    double* pair_p_vectors,int ppvi,int ppvj,
    double* pair_q_vectors,int pqvi,int pqvj,
    double* pair_p_lengths,int ppli,
    double* pair_q_lengths,int pqli,
    double* pair_health,int phi, 
    double* pair_stiffness,int psi,
    double* pair_critical_energy,int pcei,
    double* iso_pair_stiffness,int ips,
    double* iso_pair_coefficients,int ipc,
    double* iso_bending,int num_pivots, int ibjj,
    int * pair_x,int num_pairs,
    int num_broken);

void pairforces(double* sum_xforces,int lx,
    double* sum_yforces,int ly,
    double* sum_zforces,int lz,
    double* Hpq,int hi, int hj,
    double* pair_p_vectors,int ppvi,int ppvj,
    double* pair_q_vectors,int pqvi,int pqvj,
    double* pair_p_lengths,int ppli,
    double* pair_q_lengths,int pqli,
    double* pair_health,int phi, 
    double* pair_stiffness,int psi,
    int * vpt_src_a,int vsal,
    int * vpt_src_b,int vsbl,
    int * vpt_src_c,int vscl,
    double* vpt_wgt_a,int vwa,
    double* vpt_wgt_b,int vwb,
    double* vpt_wgt_c,int vwc,
    double* iso_pair_coefficients,int ipc,
    double* iso_pair_stiffness,int ips,
    double* iso_bending,int num_pivots, int ibjj,
    int * pair_x,int num_pairs,
    int * pair_p,int npp,
    int * pair_q,int npq);
void breakcoupled(
    double* bond_health,int bhi,
    double* bond_stiffness,int bsi,
    double* bond_stretches,int bs_i,
    double* bond_energies,int be_i,
    double* iso_ext_weights,int iew_i,
    double* iso_ext_stiffness,int ies_i,
    double* node_dilations, int nd_i,
    int * bond_x,int num_bonds,
    int * bond_p,int nbp,
    double* Hpq,int hi, int hj,
    double* pair_p_vectors,int ppvi,int ppvj,
    double* pair_q_vectors,int pqvi,int pqvj,
    double* pair_p_lengths,int ppli,
    double* pair_q_lengths,int pqli,
    double* pair_health,int phi, 
    double* pair_stiffness,int psi,
    double* pair_critical_energy,int pcei,
    int * bond1,int nb1,
    int * bond2,int nb2,
    double* iso_pair_stiffness,int ips,
    double* iso_pair_coefficients,int ipc,
    double* iso_bending,int num_pivots, int ibjj,
    int * pair_x,int num_pairs,
    int num_broken);
