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
#include "stdio.h"
#include "math.h"
void bondvectors(
    double* locations,int loci,int locj,
    double* bond_vectors,int bv_i,int bv_j,
    double* bond_lengths,int bl_i,
    int * bond_x,int num_bonds,
    int * bond_p,int nbp)
{
    int i,j;
    
    int x,p,q;
    
    double px,py,pz;

    for (i=0; i<num_bonds; i++)
    {
        x=bond_x[i];
        p=bond_p[i];

        px=locations[3*p]-locations[3*x];
        py=locations[3*p+1]-locations[3*x+1];
        pz=locations[3*p+2]-locations[3*x+2];
        bond_vectors[3*i]=px;
        bond_vectors[3*i+1]=py;
        bond_vectors[3*i+2]=pz;

        bond_lengths[i]=sqrt(px*px+py*py+pz*pz);

    }
}
void bondstretches(
    double* bond_vectors,int bv_i,int bv_j,
    double* bond_lengths,int bl_i,
    int * bond_x,int num_bonds,
    double* bond_ref_length,int brl_i,
    double* bond_stretches,int bs_i,
    double* iso_ext_weights,int iew_i,
    double* node_dilations, int nd_i)
{
    int i;
    int x;
    
    double stretch;
    
    for (i=0; i<nd_i; i++)
    {
        node_dilations[i]=0.0;
    }
    for (i=0; i<num_bonds; i++)
    {
        x=bond_x[i];

        stretch = bond_lengths[i]-bond_ref_length[i];
        bond_stretches[i]=stretch;
        node_dilations[x]+=stretch*iso_ext_weights[i];

    }
}
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
    int * bond_p,int nbp)
{
    int i,j;
    
    int num_real_pairs;
    int x,p,q;
    int a,b,c;
    int vpt;
    
    double px,py,pz;
    double qx,qy,qz;
    double hx,hy,hz;
    double fx,fy,fz;
    double stretch_f;

    for (i=0; i<num_bonds; i++)
    {
        x=bond_x[i];
        p=bond_p[i];


        stretch_f=bond_health[i]*(bond_stiffness[i]*bond_stretches[i]+
                iso_ext_stiffness[i]*node_dilations[x])/bond_lengths[i];
        
        px=bond_vectors[3*i];
        py=bond_vectors[3*i+1];
        pz=bond_vectors[3*i+2];

        sum_xforces[x]+=px*stretch_f;
        sum_xforces[p]-=px*stretch_f;
        
        sum_yforces[x]+=py*stretch_f;
        sum_yforces[p]-=py*stretch_f;
        
        sum_zforces[x]+=pz*stretch_f;
        sum_zforces[p]-=pz*stretch_f;
    }
}
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
    int * pair_q,int npq)
{
    int i,j;
    
    int num_real_pairs;
    int x,p,q;
    int a,b,c;
    int vpt;
    
    double px,py,pz;
    double qx,qy,qz;
    double hx,hy,hz;
    double fx,fy,fz;
    double wa,wb,wc;
    double stretch_f;
    double len_p;
    double len_q;

    num_real_pairs=num_pairs-vsal;
    
    for (i=0; i<num_real_pairs; i++)
    {
        x=pair_x[i];
        p=pair_p[i];
        q=pair_q[i];

        px=locations[3*p]-locations[3*x];
        py=locations[3*p+1]-locations[3*x+1];
        pz=locations[3*p+2]-locations[3*x+2];
        pair_p_vectors[3*i]=px;
        pair_p_vectors[3*i+1]=py;
        pair_p_vectors[3*i+2]=pz;
        len_p=sqrt(px*px+py*py+pz*pz);
        pair_p_lengths[i]=len_p;

        qx=locations[3*q]-locations[3*x];
        qy=locations[3*q+1]-locations[3*x+1];
        qz=locations[3*q+2]-locations[3*x+2];
        pair_q_vectors[3*i]=qx;
        pair_q_vectors[3*i+1]=qy;
        pair_q_vectors[3*i+2]=qz;
        len_q=sqrt(qx*qx+qy*qy+qz*qz);
        pair_q_lengths[i]=len_q;
    }
    for (i=num_real_pairs; i<num_pairs; i++)
    {
        x=pair_x[i];
        p=pair_p[i];

        vpt = i-num_real_pairs;
        px=locations[3*p]-locations[3*x];
        py=locations[3*p+1]-locations[3*x+1];
        pz=locations[3*p+2]-locations[3*x+2];
        pair_p_vectors[3*i]=px;
        pair_p_vectors[3*i+1]=py;
        pair_p_vectors[3*i+2]=pz;
        len_p=sqrt(px*px+py*py+pz*pz);
        pair_p_lengths[i]=len_p;

        a=vpt_src_a[vpt];
        b=vpt_src_b[vpt];
        c=vpt_src_c[vpt];
        wa=vpt_wgt_a[vpt];
        wb=vpt_wgt_b[vpt];
        wc=vpt_wgt_c[vpt];
        
        qx=vpt_offsets[3*vpt]+
            wa*locations[3*a]+
            wb*locations[3*b]+
            wc*locations[3*c]-locations[3*x];
        qy=vpt_offsets[3*vpt+1]+
            wa*locations[3*a+1]+
            wb*locations[3*b+1]+
            wc*locations[3*c+1]-locations[3*x+1];
        qz=vpt_offsets[3*vpt+2]+
            wa*locations[3*a+2]+
            wb*locations[3*b+2]+
            wc*locations[3*c+2]-locations[3*x+2];
        
        /*printf("x %d, %e;q %d, %e\n",x,locations[3*x+1],i,wa*locations[3*a+1]+*/
            /*wb*locations[3*b+1]+*/
            /*wc*locations[3*c+1]);*/
        /*printf("qy = %e\n",qy);*/
        
        pair_q_vectors[3*i]=qx;
        pair_q_vectors[3*i+1]=qy;
        pair_q_vectors[3*i+2]=qz;
        len_q=sqrt(qx*qx+qy*qy+qz*qz);
        pair_q_lengths[i]=len_q;
    }
}
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
    int * pair_x,int num_pairs)
{
    int i,j;
    
    int x,p,q;
    
    double px,py,pz;
    double qx,qy,qz;
    double hx,hy,hz;
    double len_p;
    double len_q;


    for (i=0; i<num_pivots;i++)
    {
        for (j=0;j<3;j++)
        {
            iso_bending[3*i+j]=0.0;
        }
    }
    for (i=0; i<num_pairs; i++)
    {
        x=pair_x[i];

        px=pair_p_vectors[3*i];
        py=pair_p_vectors[3*i+1];
        pz=pair_p_vectors[3*i+2];
        len_p=pair_p_lengths[i];
        
        qx=pair_q_vectors[3*i];
        qy=pair_q_vectors[3*i+1];
        qz=pair_q_vectors[3*i+2];
        len_q=pair_q_lengths[i];

        hx=(py*qz-qy*pz)/(len_p*len_q);
        hy=(pz*qx-qz*px)/(len_p*len_q);
        hz=(px*qy-qx*py)/(len_p*len_q);

        Hpq[3*i]=hx;
        Hpq[3*i+1]=hy;
        Hpq[3*i+2]=hz;
        /*printf("Hpq %d = %f %f %f \n",i,hx,hy,hz);*/
        iso_bending[3*x]+=iso_pair_coefficients[i]*(px+qx);
        iso_bending[3*x+1]+=iso_pair_coefficients[i]*(py+qy);
        iso_bending[3*x+2]+=iso_pair_coefficients[i]*(pz+qz);
    }
}
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
    int num_broken)
{
    int i,j;
    
    int x,p,q;
    
    double px,py,pz;
    double qx,qy,qz;
    double hx,hy,hz;
    double kx,ky,kz;
    double len_p;
    double len_q;
    double pair_energy;

    num_broken=0;

    for (i=0; i<num_pairs; i++)
    {
        x=pair_x[i];

        kx=iso_bending[3*x];
        ky=iso_bending[3*x+1];
        kz=iso_bending[3*x+2];

        px=pair_p_vectors[3*i];
        py=pair_p_vectors[3*i+1];
        pz=pair_p_vectors[3*i+2];
        len_p=pair_p_lengths[i];
        
        qx=pair_q_vectors[3*i];
        qy=pair_q_vectors[3*i+1];
        qz=pair_q_vectors[3*i+2];
        len_q=pair_q_lengths[i];

        hx=Hpq[3*i];
        hy=Hpq[3*i+1];
        hz=Hpq[3*i+2];

        pair_energy=pair_health[i]*pair_stiffness[i]*(hx*hx+hy*hy+hz*hz)/2.0;

        pair_energy+=pair_health[i]*iso_pair_stiffness[i]*kx*(px+qx);
        pair_energy+=pair_health[i]*iso_pair_stiffness[i]*ky*(py+qy);
        pair_energy+=pair_health[i]*iso_pair_stiffness[i]*kz*(pz+qz);

        if ( pair_energy > pair_critical_energy[i] ){
            pair_health[i]=0.0;
            pair_stiffness[i]=0.0;
            iso_pair_coefficients[i]=0.0;
            iso_pair_stiffness[i]=0.0;
            num_broken++;
        }
    }
}
void pairforces(
    double* sum_xforces,int lx,
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
    int * pair_q,int npq)
{
    int i,j;
    
    int num_real_pairs;
    int x,p,q;
    int a,b,c;
    int vpt;
    
    double px,py,pz;
    double qx,qy,qz;
    double hx,hy,hz;
    double fx,fy,fz;
    double kx,ky,kz;
    double wa,wb,wc;
    double stretch_f;
    double len_p;
    double len_q;

    num_real_pairs=num_pairs-vsal;

    for (i=0; i<num_real_pairs; i++)
    {
        x=pair_x[i];
        p=pair_p[i];
        q=pair_q[i];

        px=pair_p_vectors[3*i];
        py=pair_p_vectors[3*i+1];
        pz=pair_p_vectors[3*i+2];
        len_p=pair_p_lengths[i];
        
        qx=pair_q_vectors[3*i];
        qy=pair_q_vectors[3*i+1];
        qz=pair_q_vectors[3*i+2];
        len_q=pair_q_lengths[i];

        hx=Hpq[3*i];
        hy=Hpq[3*i+1];
        hz=Hpq[3*i+2];

        fx=iso_bending[3*x]*iso_pair_coefficients[i]*iso_pair_stiffness[i];
        fy=iso_bending[3*x+1]*iso_pair_coefficients[i]*iso_pair_stiffness[i];
        fz=iso_bending[3*x+2]*iso_pair_coefficients[i]*iso_pair_stiffness[i];
        
        sum_xforces[x]+=2*fx;
        sum_xforces[p]-=fx;
        sum_xforces[q]-=fx;

        sum_yforces[x]+=2*fy;
        sum_yforces[p]-=fy;
        sum_yforces[q]-=fy;

        sum_zforces[x]+=2*fz;
        sum_zforces[p]-=fz;
        sum_zforces[q]-=fz;

        fx=-(py*hz-hy*pz)*pair_stiffness[i]*pair_health[i]/(len_p*len_p);

        sum_xforces[x]+=fx;
        sum_xforces[p]-=fx;
        
        fx=(qy*hz-hy*qz)*pair_stiffness[i]*pair_health[i]/(len_q*len_q);

        sum_xforces[x]+=fx;
        sum_xforces[q]-=fx;
        
        fy=-(pz*hx-hz*px)*pair_stiffness[i]*pair_health[i]/(len_p*len_p);
        
        sum_yforces[x]+=fy;
        sum_yforces[p]-=fy;
        
        fy=(qz*hx-hz*qx)*pair_stiffness[i]*pair_health[i]/(len_q*len_q);
        
        sum_yforces[x]+=fy;
        sum_yforces[q]-=fy;
        
        fz=-(px*hy-hx*py)*pair_stiffness[i]*pair_health[i]/(len_p*len_p);
        
        sum_zforces[x]+=fz;
        sum_zforces[p]-=fz;
        
        fz=(qx*hy-hx*qy)*pair_stiffness[i]*pair_health[i]/(len_q*len_q);
        
        sum_zforces[x]+=fz;
        sum_zforces[q]-=fz;
    }
    for (i=num_real_pairs; i<num_pairs; i++)
    {
        x=pair_x[i];
        p=pair_p[i];

        vpt = i-num_real_pairs;
        px=pair_p_vectors[3*i];
        py=pair_p_vectors[3*i+1];
        pz=pair_p_vectors[3*i+2];
        len_p=pair_p_lengths[i];
        
        qx=pair_q_vectors[3*i];
        qy=pair_q_vectors[3*i+1];
        qx=pair_q_vectors[3*i+2];
        len_q=pair_q_lengths[i];

        hx=Hpq[3*i];
        hy=Hpq[3*i+1];
        hz=Hpq[3*i+2];


        a=vpt_src_a[vpt];
        b=vpt_src_b[vpt];
        c=vpt_src_c[vpt];
        wa=vpt_wgt_a[vpt];
        wb=vpt_wgt_b[vpt];
        wc=vpt_wgt_c[vpt];

        fx=iso_bending[3*x]*iso_pair_coefficients[i]*iso_pair_stiffness[i];
        fy=iso_bending[3*x+1]*iso_pair_coefficients[i]*iso_pair_stiffness[i];
        fz=iso_bending[3*x+2]*iso_pair_coefficients[i]*iso_pair_stiffness[i];
        
        sum_xforces[x]+=2*fx;
        sum_xforces[p]-=fx;
        sum_xforces[a]-=wa*fx;
        sum_xforces[b]-=wb*fx;
        sum_xforces[c]-=wc*fx;

        sum_yforces[x]+=2*fy;
        sum_yforces[p]-=fy;
        sum_yforces[a]-=wa*fy;
        sum_yforces[b]-=wb*fy;
        sum_yforces[c]-=wc*fy;

        sum_zforces[x]+=2*fz;
        sum_zforces[p]-=fz;
        sum_zforces[a]-=wa*fz;
        sum_zforces[b]-=wb*fz;
        sum_zforces[c]-=wc*fz;

        fx=-(py*hz-hy*pz)*pair_stiffness[i]*pair_health[i]/(len_p*len_p);

        sum_xforces[x]+=fx;
        sum_xforces[p]-=fx;
        
        fx=(qy*hz-hy*qz)*pair_stiffness[i]*pair_health[i]/(len_q*len_q);

        sum_xforces[x]+=fx;
        sum_xforces[a]-=wa*fx;
        sum_xforces[b]-=wb*fx;
        sum_xforces[c]-=wc*fx;
        
        fy=-(pz*hx-hz*px)*pair_stiffness[i]*pair_health[i]/(len_p*len_p);
        
        sum_yforces[x]+=fy;
        sum_yforces[p]-=fy;
        
        fy=(qz*hx-hz*qx)*pair_stiffness[i]*pair_health[i]/(len_q*len_q);
        
        sum_yforces[x]+=fy;
        sum_yforces[a]-=wa*fy;
        sum_yforces[b]-=wb*fy;
        sum_yforces[c]-=wc*fy;
        
        fz=-(px*hy-hx*py)*pair_stiffness[i]*pair_health[i]/(len_p*len_p);
        
        sum_zforces[x]+=fz;
        sum_zforces[p]-=fz;
        
        fz=(qx*hy-hx*qy)*pair_stiffness[i]*pair_health[i]/(len_q*len_q);
        
        sum_zforces[x]+=fz;
        sum_zforces[a]-=wa*fz;
        sum_zforces[b]-=wb*fz;
        sum_zforces[c]-=wc*fz;
    }
}
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
    int num_broken)
{
    int i,j;
    
    int num_real_pairs;
    int x,p,q;
    
    double px,py,pz;
    double qx,qy,qz;
    double hx,hy,hz;
    double kx,ky,kz;
    double len_p;
    double len_q;
    double pair_energy;

    num_real_pairs=nb2;
    num_broken=0;
    for (i=0; i<num_bonds; i++)
    {
        x=bond_x[i];

        bond_energies[i]=bond_health[i]*(bond_stiffness[i]*bond_stretches[i]*bond_stretches[i]+
                iso_ext_stiffness[i]*node_dilations[x]*node_dilations[x]);
        
    }

    for (i=0; i<num_pairs; i++)
    {
        x=pair_x[i];

        kx=iso_bending[3*x];
        ky=iso_bending[3*x+1];
        kz=iso_bending[3*x+2];

        px=pair_p_vectors[3*i];
        py=pair_p_vectors[3*i+1];
        pz=pair_p_vectors[3*i+2];
        
        qx=pair_q_vectors[3*i];
        qy=pair_q_vectors[3*i+1];
        qz=pair_q_vectors[3*i+2];

        hx=Hpq[3*i];
        hy=Hpq[3*i+1];
        hz=Hpq[3*i+2];

        pair_energy=pair_health[i]*pair_stiffness[i]*(hx*hx+hy*hy+hz*hz)/2.0;

        pair_energy+=pair_health[i]*iso_pair_stiffness[i]*kx*(px+qx);
        pair_energy+=pair_health[i]*iso_pair_stiffness[i]*ky*(py+qy);
        pair_energy+=pair_health[i]*iso_pair_stiffness[i]*kz*(pz+qz);
        pair_energy+=bond_energies[bond1[i]];
        if ( i<num_real_pairs ){
            pair_energy+=bond_energies[bond2[i]];
        }


        if ( pair_energy > pair_critical_energy[i] ){
            pair_health[i]=0.0;
            pair_stiffness[i]=0.0;
            iso_pair_coefficients[i]=0.0;
            iso_pair_stiffness[i]=0.0;
            num_broken++;
            bond_health[bond1[i]]=0.0;
            if ( i<num_real_pairs ){
                bond_health[bond2[i]]=0.0;
            }
        }
    }
}
