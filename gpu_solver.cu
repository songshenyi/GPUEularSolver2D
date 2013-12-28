
#include "gpu_fvm.cuh"
#include "cuPrintf.cu"



	double EPSM;
	double *deltT;
	double drou,dt,*rou;
	RESD_S	Res_S;
	W 		*w0;
	W      *Wallw,*Farw;
	
	W_S    w_S,w0_S, Wallw_S, Farw_S;

	double	*d_deltT;
	double	*d_drou;
	W		*d_w0;
	RESD_S	d_Res_S, d_R1_S;
	W		*d_Wallw, *d_Farw;
	
	W_S		d_w_S, d_w0_S, d_Wallw_S, d_Farw_S;
	
	EDGE_S 	edge_S,d_edge_S;
	
	NODE_S	node_S, d_node_S;
	CELL_S	cell_S, d_cell_S;
	
	DT_E 	*d_deltT_e;
	DT_E_S	d_deltT_e_S;
	DR_E	*d_Res_e;
	DR_E_S	d_Res_e_S;

#define MaxThreads 64

__global__ void kernel_RK_Init(double *deltT, RESD_S Res_S, int Ncell)
{
	int id=blockDim.x * blockIdx.x + threadIdx.x;
	if(id>0 && id<=Ncell)
	{
		deltT[id] = 0.0;
		Res_S.data[0][id] = 0.0;
		Res_S.data[1][id] = 0.0;
		Res_S.data[2][id] = 0.0;
		Res_S.data[3][id] = 0.0;
	}		
}

__forceinline__ __device__ void kernel_CalculationPTA(W &WW, double GAMA, double R)
{
	double U,V,E;
	
	U=WW.density_U/WW.density;
	V=WW.density_V/WW.density;
	E=WW.density_E/WW.density-0.5*(U*U+V*V);
	
	WW.P=(GAMA-1.0)*WW.density*E;
	WW.A=sqrt(GAMA*WW.P/WW.density);
	WW.T=(GAMA-1.0)*E/R;
}

__forceinline__ __device__ void kernel_Far_Boundary(W &w0, W DW, double vectorx, double vectory, 
							double GAMA, double UI, double VI,
							double AIN, double TIN, double VIN,
							double R, double PIS)
{
	double PB,PD,PA,nx,ny,nn,Un,Uncell,AD,ROUD,UB,VB,UD,VD;   //B-boundary,D-domain,A-Initial variable

	nn=sqrt(vectorx*vectorx+vectory*vectory);
	nx=vectory/nn;   //gradient
	ny=vectorx/nn;
	
	UD=DW.density_U/DW.density;
	VD=DW.density_V/DW.density;

	Un=fabs(UI*nx+VI*ny);		//contravariant velocity of the coming flow
	Uncell=UD*nx+VD*ny;			//contravariant velocity of the cell's flow

	if(Un/AIN>1.0)   //supersonic
	{
		if(Uncell>0.0)			//outflow
			w0=DW;

		if(Uncell<=0.0)			//infow
		{
			w0.density  =1.0;
			w0.density_U=UI;
			w0.density_V=VI;
			w0.density_E=R*TIN/(VIN*VIN)/(GAMA-1.0)+0.50;

			kernel_CalculationPTA(w0, GAMA,R);	
		}		
	}

	else		       //subsonic
	{
		PA	=PIS;
		PD	=DW.P;
		AD	=DW.A;
		ROUD=DW.density;

		if(Uncell>0.0)//outflow
		{
			PB=PA;
			w0.P=PB;
			w0.density=ROUD+(PB-PD)/AD/AD;

			UB=UD+nx*(PD-PB)/ROUD/AD;
			VB=VD+ny*(PD-PB)/ROUD/AD;

			w0.density_U=UB*w0.density;
			w0.density_V=VB*w0.density;
			w0.density_E=PB/(GAMA-1.0)+0.5*(UB*UB+VB*VB)*w0.density;

			kernel_CalculationPTA(w0, GAMA,R);	
		}

		if(Uncell<=0.0)//inflow
		{
			PB=0.5*( PA+PD- ROUD*AD* ( nx*(UI-UD)+ny*(VI-VD) ) );
			w0.P=PB;
			
			w0.density=1.0+(PB-PA)/AD/AD;
			
			UB=UI-nx*(PA-PB)/ROUD/AD;
			VB=VI-ny*(PA-PB)/ROUD/AD;

			w0.density_U=UB*w0.density;
			w0.density_V=VB*w0.density;
			w0.density_E=PB/(GAMA-1.0)+0.5*(UB*UB+VB*VB)*w0.density;

			kernel_CalculationPTA(w0,GAMA,R);	
		}
	}	
}

__global__ void kernel_Boundary(W_S w_S, W_S Wallw_S, 
							W_S Farw_S, CELL* __restrict__ cell, 
							EDGE_S edge_S, double GAMA, double UI,
							double VI, double AIN, double TIN,
							double VIN,	double R, double PIS,
							int Nedge)
{
	int wallid, farid;
	double	UU,VV,UV;
	double	nij[2];
	int id=blockDim.x * blockIdx.x + threadIdx.x;
	W	 w0,wd;
	int ELog, left_cell;
	double vectorx, vectory,vectorn ;
	if(id>0 && id<=Nedge)
	{
		//	Iedge = edge[id];
		ELog = edge_S.ELog[id];
		left_cell = edge_S.left_cell[id];
		vectorx = edge_S.vectorx[id];
		vectory = edge_S.vectory[id];
		vectorn = edge_S.vectorn[id];
			
		if(ELog==1) // FarFeild
		{
			wd.density=w_S.density[left_cell];
			wd.density_U=w_S.density_U[left_cell];
			wd.density_V=w_S.density_V[left_cell];
			wd.density_E=w_S.density_E[left_cell];
			wd.P=w_S.P[left_cell];
			wd.T=w_S.T[left_cell];
			wd.A=w_S.A[left_cell];
			
			farid=edge_S.farfieldid[id];
//			w0=Farw[farid];

			kernel_Far_Boundary( w0 ,wd, 
						vectorx,vectory,GAMA,UI,VI,AIN,TIN,VIN,R,PIS);
						
			Farw_S.density[farid] = w0.density;
			Farw_S.density_U[farid] = w0.density_U;
			Farw_S.density_V[farid] = w0.density_V;
			Farw_S.density_E[farid] = w0.density_E;
			Farw_S.P[farid] = w0.P;
			Farw_S.T[farid] = w0.T;
			Farw_S.A[farid] = w0.A;
			
//			Farw[farid]=w0;
//			w[Iedge.left_cell]=wd;			
		}
		
		if(ELog==2) //Wall
		{ 
			wd.density=w_S.density[left_cell];
			wd.density_U=w_S.density_U[left_cell];
			wd.density_V=w_S.density_V[left_cell];
			wd.density_E=w_S.density_E[left_cell];
			wd.P=w_S.P[left_cell];
			wd.T=w_S.T[left_cell];
			wd.A=w_S.A[left_cell];
						
			nij[0]=vectory/vectorn;
			nij[1]=vectorx/vectorn;

			UU=wd.density_U/wd.density;
			VV=wd.density_V/wd.density;
			UV=UU*nij[0]+VV*nij[1];

			UU=UU-2*UV*nij[0];
			VV=VV-2*UV*nij[1];
		
			wallid=edge_S.wallid[id];

			wd.density_U=wd.density*UU;
			wd.density_V=wd.density*VV;
		
			Wallw_S.density[wallid]=wd.density;
			Wallw_S.density_U[wallid]=wd.density_U;
			Wallw_S.density_V[wallid]=wd.density_V;
			Wallw_S.density_E[wallid]=wd.density_E;
			Wallw_S.P[wallid]=wd.P;
			Wallw_S.T[wallid]=wd.T;
			Wallw_S.A[wallid]=wd.A;
//			cuPrintf("%d %d %d %f %f\n",id, wallid, edge[id].left_cell,Wallw[wallid].density_U,Wallw[wallid].density_V);			
		}
	}
}

__global__ void kernel_Gradient(W_S w_S, W_S Wallw_S, W_S Farw_S, CELL* __restrict__ cell,
				EDGE_S  edge_S,double* __restrict__ CellArea,NODE_S node_S, int Ncell)
{
	
	int i,id,tid;
	double deltU[2][4];	
	int j,N1,N2;
//	__shared__ double	nij[MaxThreads][2], ds[MaxThreads], area[MaxThreads];
	double area;
	__shared__ double Ujmax[MaxThreads][4],Ujmin[MaxThreads][4];
	double comp[4];
	__shared__ double	vectorx[MaxThreads],vectory[MaxThreads];
	__shared__ double density[MaxThreads], density_U[MaxThreads], density_V[MaxThreads], density_E[MaxThreads];
//	W	  ghost;
//	Neighbor neighbor;
	int  celledge,neicell;
	
	tid = threadIdx.x;
	int icell=blockDim.x * blockIdx.x + threadIdx.x;
	if(icell>0 && icell<=Ncell)
	{
		density[tid]   = w_S.density[icell];
        density_U[tid] = w_S.density_U[icell];
        density_V[tid] = w_S.density_V[icell];
        density_E[tid] = w_S.density_E[icell];

		for(j=0;j<2;j++)
		{		
			deltU[j][0]=0.0;
			deltU[j][1]=0.0;
			deltU[j][2]=0.0;
			deltU[j][3]=0.0;
		}
		
		for(i=0;i<3;i++)
		{
			N1=cell[icell].Point[(i+1)%3];
			N2=cell[icell].Point[(i+2)%3];
			
			vectorx[tid]=node_S.x[N1]-node_S.x[N2];    //2->3,3->1,1->2
			vectory[tid]=node_S.y[N2]-node_S.y[N1];
			
//			for(j=0;j<3;j++)  //find the neighbor cell connecting edge
//			{		
				celledge = cell[icell].neighbor[i].celledge;		
				neicell  = cell[icell].neighbor[i].neicell;	
//				NN1=edge[celledge].node1;
//				NN2=edge[celledge].node2;
//				if( (NN1==N1&&NN2==N2) || (NN1==N2&&NN2==N1) )
//				{
//					break;
//				}
//			}
			
			
//			neigh =neighbor.neicell;					    //the i-th neighbor's cell number
//			ds[tid]    =edge[celledge].vectorn;	//edge with the i-th neighbor's cell 
//			nij[tid][0]=vectory[tid]/ds[tid];
//			nij[tid][1]=vectorx[tid]/ds[tid];
			
			if(neicell >0)//domain
			{   
				comp[0]=w_S.density[neicell];
				comp[1]=w_S.density_U[neicell];
				comp[2]=w_S.density_V[neicell];
				comp[3]=w_S.density_E[neicell];
			}
//			else
//			{
				if(neicell==-1) // farfield
				{
					id=edge_S.farfieldid[celledge];
					comp[0]=Farw_S.density[id];
					comp[1]=Farw_S.density_U[id];
					comp[2]=Farw_S.density_V[id];
					comp[3]=Farw_S.density_E[id];
				}
								
				if(neicell==-2) //wall
				{
					id=edge_S.wallid[celledge];
					comp[0]=Wallw_S.density[id];
					comp[1]=Wallw_S.density_U[id];
					comp[2]=Wallw_S.density_V[id];
					comp[3]=Wallw_S.density_E[id];					
				}

//			}
			
			
			if(i==0)
			{     													//search the Ujmin/max
				for(j=0;j<4;j++)
				{
					Ujmax[tid][j]=comp[j];
					Ujmin[tid][j]=comp[j];
				}
			}
			else
			{
				for(j=0;j<4;j++)
				{
//					Ujmin[tid][j] = fmin(Ujmin[tid][j], comp[j]);
//					Ujmax[tid][j] = fmax(Ujmax[tid][j], comp[j]);
					if(comp[j]<Ujmin[tid][j])
						Ujmin[tid][j]=comp[j];
					if(comp[j]>Ujmax[tid][j])
						Ujmax[tid][j]=comp[j];
				}
			}
			
			
//			for(j=0;j<2;j++)
			{											//j=0 for x direct j=1 for y direct 
				deltU[0][0]+=0.5*(density[tid]   + comp[0] )*vectory[tid];
				deltU[0][1]+=0.5*(density_U[tid] + comp[1] )*vectory[tid];
				deltU[0][2]+=0.5*(density_V[tid] + comp[2] )*vectory[tid];
				deltU[0][3]+=0.5*(density_E[tid] + comp[3] )*vectory[tid];
				deltU[1][0]+=0.5*(density[tid]   + comp[0] )*vectorx[tid];
				deltU[1][1]+=0.5*(density_U[tid] + comp[1] )*vectorx[tid];
				deltU[1][2]+=0.5*(density_V[tid] + comp[2] )*vectorx[tid];
				deltU[1][3]+=0.5*(density_E[tid] + comp[3] )*vectorx[tid];
			}

		}
		

		area=1.0/CellArea[icell];
		for(j=0;j<2;j++)
		{
			cell[icell].deltU[j][0] = deltU[j][0]*area;
			cell[icell].deltU[j][1] = deltU[j][1]*area;
			cell[icell].deltU[j][2] = deltU[j][2]*area;
			cell[icell].deltU[j][3] = deltU[j][3]*area;
		}

	
		comp[0]=density[tid];		//Ui
		comp[1]=density_U[tid];
		comp[2]=density_V[tid];
		comp[3]=density_E[tid];

		for(j=0;j<4;j++)
		{							//search the Umin/max
/*			if(comp[j]>Ujmax[j])
				cell[icell].Umax[j]=comp[j];
			else
				cell[icell].Umax[j]=Ujmax[j];
			if(comp[j]<Ujmin[j])
				cell[icell].Umin[j]=comp[j];
			else
				cell[icell].Umin[j]=Ujmin[j];
*/
			cell[icell].Umax[j] = fmax(comp[j], Ujmax[tid][j]);
			cell[icell].Umin[j] = fmin(comp[j], Ujmin[tid][j]);
		}
	}
	
}

__global__ void kernel_deltT(W_S w_S, W_S Wallw_S, W_S Farw_S, 
							DT_E_S deltT_e_S, EDGE_S edge_S,
							double GAMA, int Nedge)
{
//	EDGE   Iedge;
//	W 	   RW;
	int iedge=blockDim.x * blockIdx.x + threadIdx.x;
	double u,v,rou,p,VC,CS,SR,EE;
	int    ELog, left_cell,right_cell, farfieldid, wallid;
	double  vectorx, vectory;
	double  RW_density, RW_density_U, RW_density_V, RW_density_E;
	
	if(iedge>0 && iedge<=Nedge)
	{
		ELog = edge_S.ELog[iedge];
		left_cell = edge_S.left_cell[iedge];
		right_cell = edge_S.right_cell[iedge];
		farfieldid = edge_S.farfieldid[iedge];
		wallid = edge_S.wallid[iedge];
		vectorx = edge_S.vectorx[iedge];
		vectory = edge_S.vectory[iedge];
		
		if(ELog==0)//inside the fluid field
		{
//			LW = w[Iedge.left_cell];
			RW_density = w_S.density[right_cell];
			RW_density_U = w_S.density_U[right_cell];
			RW_density_V = w_S.density_V[right_cell];
			RW_density_E = w_S.density_E[right_cell];
		}
				
		if(ELog==1)   //FarField
	    {
//	    	LW=w[Iedge.left_cell];
			RW_density=Farw_S.density[farfieldid];
			RW_density_U=Farw_S.density_U[farfieldid];
			RW_density_V=Farw_S.density_V[farfieldid];
			RW_density_E=Farw_S.density_E[farfieldid];
		}

    	if(ELog==2)  //Wall
		{
//			LW=w[Iedge.left_cell];
			RW_density=Wallw_S.density[wallid];
			RW_density_U=Wallw_S.density_U[wallid];
			RW_density_V=Wallw_S.density_V[wallid];
			RW_density_E=Wallw_S.density_E[wallid];
		}
							
		rou=(w_S.density[left_cell]+RW_density)*0.5;   //the average value of the left cell & the right cell to an edge
		u=(w_S.density_U[left_cell]+RW_density_U)/rou*0.5;
		v=(w_S.density_V[left_cell]+RW_density_V)/rou*0.5;
		EE=(w_S.density_E[left_cell]+RW_density_E)*0.5/rou-(u*u+v*v)*0.5;

		VC=u*vectory+v*vectorx;	
		p=rou*EE*(GAMA-1.0);  //pressure
		CS=sqrt((GAMA*p/rou)*(vectorx*vectorx+vectory*vectory));
		SR=fabs(VC)+CS; // 这一句可以省去
			
	
			
		deltT_e_S.left[iedge]=SR; 
		
		if(right_cell!=-1)
		{
			deltT_e_S.right[iedge]=SR;
		}
	}
		
}

__forceinline__ __device__ void kernel_Limiter(
								double density,
								double density_U,
								double density_V,
								double density_E,
								int Cellnum,	
									CELL* __restrict__ cell,
								 double delt2[4])
							//	double &delt2_0,double &delt2_1,
							//	double &delt2_2,double &delt2_3)
{
	double UI[4];
//	double delt2[4];
	int i;
	double fi;

//	r[0]=Iedge.midx-cell[Iedge.left_cell].center[0];
//	r[1]=Iedge.midy-cell[Iedge.left_cell].center[1];

	UI[0]=density;
	UI[1]=density_U;
	UI[2]=density_V;
	UI[3]=density_E;

	for(i=0;i<4;i++)
	{
		if(delt2[i] > 0 )
			fi = fmin(( cell[Cellnum].Umax[i]-UI[i] )/delt2[i] , 1.0);

		if(delt2[i] < 0 )
			fi = fmin(( cell[Cellnum].Umin[i]-UI[i] )/delt2[i] , 1.0);
			
		if(delt2[i] ==0)
			fi = 1.0 ;
		
		delt2[i]*=fi;
	}

} 

__forceinline__ __device__ void kernel_LBJinterpolate(W_S w_S, 
									int left_cell, int right_cell, 
									double midx, double midy, CELL* __restrict__ cell, 
									W &LW, W &RW, double* __restrict__ CellArea)
{
	double density,density_U,density_V,density_E;
	double r[2];
//	double delt2_0,delt2_1,delt2_2,delt2_3;
	double	delt2[4];
	int cell_id;	
	
//	id=threadIdx.x;
	cell_id=left_cell;
	density=w_S.density[cell_id];
	density_U=w_S.density_U[cell_id];
	density_V=w_S.density_V[cell_id];
	density_E=w_S.density_E[cell_id];
	
	r[0]=midx-cell[cell_id].center[0];
	r[1]=midy-cell[cell_id].center[1];
	
	delt2[0]=cell[cell_id].deltU[0][0]*r[0]+cell[cell_id].deltU[1][0]*r[1];
	delt2[1]=cell[cell_id].deltU[0][1]*r[0]+cell[cell_id].deltU[1][1]*r[1];
	delt2[2]=cell[cell_id].deltU[0][2]*r[0]+cell[cell_id].deltU[1][2]*r[1];
	delt2[3]=cell[cell_id].deltU[0][3]*r[0]+cell[cell_id].deltU[1][3]*r[1];
	
	kernel_Limiter(density,density_U,density_V,density_E,cell_id,cell,delt2);//_0,delt2_1,delt2_2,delt2_3);
	
	LW.density  =density  +delt2[0];
	LW.density_U=density_U+delt2[1];
	LW.density_V=density_V+delt2[2];
	LW.density_E=density_E+delt2[3];
	
	cell_id=right_cell;
	density=w_S.density[cell_id];
	density_U=w_S.density_U[cell_id];
	density_V=w_S.density_V[cell_id];
	density_E=w_S.density_E[cell_id];
	
	r[0]=midx-cell[cell_id].center[0];
	r[1]=midy-cell[cell_id].center[1];	
	
	delt2[0]=cell[cell_id].deltU[0][0]*r[0]+cell[cell_id].deltU[1][0]*r[1];
	delt2[1]=cell[cell_id].deltU[0][1]*r[0]+cell[cell_id].deltU[1][1]*r[1];
	delt2[2]=cell[cell_id].deltU[0][2]*r[0]+cell[cell_id].deltU[1][2]*r[1];
	delt2[3]=cell[cell_id].deltU[0][3]*r[0]+cell[cell_id].deltU[1][3]*r[1];
	
	kernel_Limiter(density,density_U,density_V,density_E,cell_id,cell,delt2);//_0,delt2_1,delt2_2,delt2_3);
	
	RW.density  =density  +delt2[0];
	RW.density_U=density_U+delt2[1];
	RW.density_V=density_V+delt2[2];
	RW.density_E=density_E+delt2[3];	
}


__inline__ __device__ void kernel_Calc_Function(W &w,double &A,double &U,
							double &V,double &H,double &P,
							double GAMA)
{
//	double E;
	double temp;
	temp=1.0/w.density;
	U=w.density_U*temp;
	V=w.density_V*temp;
//	E=w.density_E/w.density-0.5*(U*U+V*V);

	P=(GAMA-1.0)*w.density*(w.density_E*temp-0.5*(U*U+V*V));
	A=sqrt(GAMA*P*temp);
	H=w.density_E*temp+P*temp;
}


__forceinline__ __device__ void kernel_ROE_SCHEME(W &LW,W &RW,double nx,
				double ny,double GAMA,double FLUX[4])
{
	double ROUB,UB,VB,HB,QB,VC,AB,DVC,epsilon;
	double A[2],U[2],V[2],H[2],P[2],M[2],FF[3][4],AD[3],WL[4],WR[4];
	double lrou,rrou, sqrtlrou,sqrtrrou;

	kernel_Calc_Function(LW,A[0],U[0],V[0],H[0],P[0],GAMA);
	kernel_Calc_Function(RW,A[1],U[1],V[1],H[1],P[1],GAMA);
	

	M[0]=(U[0]*nx+V[0]*ny);	
	M[1]=(U[1]*nx+V[1]*ny);	

	DVC=M[1]-M[0];

	lrou=LW.density;
	rrou=RW.density;
	sqrtlrou = sqrt(lrou);
	sqrtrrou = sqrt(rrou);
    
	ROUB=sqrtlrou * sqrtrrou;

	UB=( U[0] * sqrtlrou +U[1] * sqrtrrou )	/(sqrtlrou + sqrtrrou);

	VB=( V[0] * sqrtlrou +V[1] * sqrtrrou )	/(sqrtlrou + sqrtrrou);

	HB=( H[0] * sqrtlrou +H[1] * sqrtrrou )	/(sqrtlrou + sqrtrrou);


	QB=UB*UB+VB*VB;
	AB=sqrt( (GAMA-1.0) * (HB-QB/2.0) );
	VC=UB*nx+VB*ny;  
	
	epsilon=0.05;

	if(fabs(VC-AB)<epsilon)
		AD[0]=((VC-AB)*(VC-AB)+epsilon*epsilon)/(2*epsilon);
	else
		AD[0]=fabs(VC-AB);

	if(fabs(VC)<epsilon)
		AD[1]=(VC*VC+epsilon*epsilon)/(2*epsilon);
	else
		AD[1]=fabs(VC);

	if(fabs(VC+AB)<epsilon)
		AD[2]=((VC+AB)*(VC+AB)+epsilon*epsilon)/(2*epsilon);
	else
		AD[2]=fabs(VC+AB);

	FF[0][0]=AD[0]*((P[1]-P[0])-ROUB*AB*DVC)/(2.0*AB*AB);
	FF[0][1]=FF[0][0]*(UB-AB*nx);
	FF[0][2]=FF[0][0]*(VB-AB*ny);
	FF[0][3]=FF[0][0]*(HB-AB*VC);
	
	FF[1][0]=AD[1]*((rrou-lrou)-(P[1]-P[0])/(AB*AB));
	FF[1][1]=FF[1][0]*UB+AD[1]*ROUB*(U[1]-U[0]-DVC*nx);
	FF[1][2]=FF[1][0]*VB+AD[1]*ROUB*(V[1]-V[0]-DVC*ny);
	FF[1][3]=FF[1][0]*QB*0.5+AD[1]*ROUB*(UB*(U[1]-U[0])+VB*(V[1]-V[0])-DVC*VC);
	
	FF[2][0]=AD[2]*((P[1]-P[0])+ROUB*AB*DVC)/(2.0*AB*AB);
	FF[2][1]=FF[2][0]*(UB+AB*nx);
	FF[2][2]=FF[2][0]*(VB+AB*ny);
	FF[2][3]=FF[2][0]*(HB+AB*VC);

	WL[0]=lrou*M[0];
	WL[1]=U[0]*WL[0]+nx*P[0];
	WL[2]=V[0]*WL[0]+ny*P[0];
	WL[3]=WL[0]*H[0];
	
	WR[0]=rrou*M[1];
	WR[1]=U[1]*WR[0]+nx*P[1];
	WR[2]=V[1]*WR[0]+ny*P[1];
	WR[3]=WR[0]*H[1];

	for(int k=0;k<4;k++)		//ROE scheme
		FLUX[k]=0.5*((WL[k]+WR[k])-(FF[0][k]+FF[1][k]+FF[2][k]));
}



__global__ void kernel_Flux2(W_S w_S, W_S Wallw_S, W_S Farw_S, double* __restrict__ CellArea, 
							EDGE_S edge_S, CELL* __restrict__ cell,
							DR_E_S Res_e_S, double GAMA, int Nedge)	
{
//	double	ds;
	double FLUX[4];
	double nn;
//	double nx,ny;
	__shared__ W 	   LW[MaxThreads],RW[MaxThreads];//,LW1,RW1;
//	EDGE   Iedge;	
//	int	   i;
	int  ELog, left_cell, right_cell, farfieldid, wallid;
	double  vectorx, vectory, vectorn, midx, midy;
	
	int 	tid = threadIdx.x;
	int iedge=blockDim.x * blockIdx.x + threadIdx.x;
	if(iedge>0 && iedge<=Nedge)
	{
//		Iedge = edge[iedge];
		ELog = edge_S.ELog[iedge];
		vectorx = edge_S.vectorx[iedge];
		vectory = edge_S.vectory[iedge];
		vectorn = edge_S.vectorn[iedge];
		left_cell = edge_S.left_cell[iedge];
		right_cell = edge_S.right_cell[iedge];
		farfieldid = edge_S.farfieldid[iedge];
		wallid = edge_S.wallid[iedge];
		midx = edge_S.midx[iedge];
		midy = edge_S.midy[iedge];
//		ds=sqrt(Iedge.vectorx*Iedge.vectorx+Iedge.vectory*Iedge.vectory);
//		nn=sqrt(vectorx*vectorx+Iedge.vectory*Iedge.vectory);

		
		if(ELog==0)//inside the fluid field
		{
			kernel_LBJinterpolate(w_S,left_cell, right_cell, midx, midy,cell,LW[tid],RW[tid],CellArea);
		}
		
		if(ELog==1)   //FarField
	    {
	    	LW[tid].density=w_S.density[left_cell];
	    	LW[tid].density_U=w_S.density_U[left_cell];
	    	LW[tid].density_V=w_S.density_V[left_cell];
	    	LW[tid].density_E=w_S.density_E[left_cell];
	    	LW[tid].P=w_S.P[left_cell];
	    	LW[tid].T=w_S.T[left_cell];
	    	LW[tid].A=w_S.A[left_cell];

	    	RW[tid].density=Farw_S.density[farfieldid];
	    	RW[tid].density_U=Farw_S.density_U[farfieldid];
	    	RW[tid].density_V=Farw_S.density_V[farfieldid];
	    	RW[tid].density_E=Farw_S.density_E[farfieldid];
	    	RW[tid].P=Farw_S.P[farfieldid];
	    	RW[tid].T=Farw_S.T[farfieldid];
	    	RW[tid].A=Farw_S.A[farfieldid];			
		}

    	if(ELog==2)  //Wall
		{
			LW[tid].density=w_S.density[left_cell];
	    	LW[tid].density_U=w_S.density_U[left_cell];
	    	LW[tid].density_V=w_S.density_V[left_cell];
	    	LW[tid].density_E=w_S.density_E[left_cell];
	    	LW[tid].P=w_S.P[left_cell];
	    	LW[tid].T=w_S.T[left_cell];
	    	LW[tid].A=w_S.A[left_cell];

	    	RW[tid].density=Wallw_S.density[wallid];
	    	RW[tid].density_U=Wallw_S.density_U[wallid];
	    	RW[tid].density_V=Wallw_S.density_V[wallid];
	    	RW[tid].density_E=Wallw_S.density_E[wallid];
	    	RW[tid].P=Wallw_S.P[wallid];
	    	RW[tid].T=Wallw_S.T[wallid];
	    	RW[tid].A=Wallw_S.A[wallid];		
		}					

		kernel_ROE_SCHEME(LW[tid],RW[tid],vectory/vectorn,vectorx/vectorn,GAMA,FLUX);

//		if(Iedge.ELog==0)//inside the fluid field
//		{
//			LW = w[Iedge.left_cell];
//			RW = w[Iedge.right_cell];
//		}

//		deltT_e[iedge].left=kernel_Calc_SRI(LW,RW,Iedge,GAMA); 


		FLUX[0]*=vectorn;
		FLUX[1]*=vectorn;
		FLUX[2]*=vectorn;
		FLUX[3]*=vectorn;
		
		Res_e_S.left[0][iedge]=FLUX[0];
		Res_e_S.left[1][iedge]=FLUX[1];
		Res_e_S.left[2][iedge]=FLUX[2];
		Res_e_S.left[3][iedge]=FLUX[3];

		if(right_cell!=-1)
		{
			Res_e_S.right[0][iedge]=FLUX[0];
			Res_e_S.right[1][iedge]=FLUX[1];
			Res_e_S.right[2][iedge]=FLUX[2];
			Res_e_S.right[3][iedge]=FLUX[3];
//			deltT_e[iedge].right=deltT_e[iedge].left;
		}
		else
		{
			Res_e_S.right[0][iedge]=0;
			Res_e_S.right[1][iedge]=0;
			Res_e_S.right[2][iedge]=0;
			Res_e_S.right[3][iedge]=0;			
		}
	}
}



__global__ void kernel_Flux_e2c(DT_E_S deltT_e_S, double* __restrict__ deltT,
						DR_E_S Res_e_S, RESD_S Res_S,
						CELL* __restrict__ cell, EDGE_S edge_S, double* __restrict__ CellArea,
								double CFL,int Ncell)
{	
	int i,j,iedge[3];
//	EDGE e;
	double T=0;
	RESD Res_data;
//	DR_E Res_data_e;
//	__shared__ double dt_s[MaxThreads];
//	int	blockSize = blockDim.x;	
//	int	tid = threadIdx.x;
	int left_cell, right_cell;
	
	Res_data.data[0]=0.0;
	Res_data.data[1]=0.0;
	Res_data.data[2]=0.0;
	Res_data.data[3]=0.0;
		
	int icell=blockDim.x * blockIdx.x + threadIdx.x;
	if(icell>0 && icell<=Ncell)
	{
//		n=cell[icell].edgeNum;
		iedge[0]=cell[icell].neighbor[0].celledge;
		iedge[1]=cell[icell].neighbor[1].celledge;
		iedge[2]=cell[icell].neighbor[2].celledge;

//		n=Icell.edgeNum;
	
		for(i=0; i<3; i++)
		{
//			iedge=Icell.edge[i];
//			e=edge[iedge[i]];
/*			Res_data_e.left[0]=Res_e_S.left[0][iedge[i]];
			Res_data_e.left[1]=Res_e_S.left[1][iedge[i]];
			Res_data_e.left[2]=Res_e_S.left[2][iedge[i]];
			Res_data_e.left[3]=Res_e_S.left[3][iedge[i]];
			Res_data_e.right[0]=Res_e_S.right[0][iedge[i]];
			Res_data_e.right[1]=Res_e_S.right[1][iedge[i]];
			Res_data_e.right[2]=Res_e_S.right[2][iedge[i]];
			Res_data_e.right[3]=Res_e_S.right[3][iedge[i]]; */
			if(edge_S.left_cell[iedge[i]] == icell)
			{
				T+=deltT_e_S.left[iedge[i]];
				
				for(j=0;j<4;j++)
					Res_data.data[j]+=Res_e_S.left[j][iedge[i]];
			}
			if(edge_S.right_cell[iedge[i]] == icell)
			{
				T+=deltT_e_S.right[iedge[i]];

				for(j=0;j<4;j++)
					Res_data.data[j]-=Res_e_S.right[j][iedge[i]];

			}
		}
		
		Res_S.data[0][icell]=Res_data.data[0];
		Res_S.data[1][icell]=Res_data.data[1];
		Res_S.data[2][icell]=Res_data.data[2];
		Res_S.data[3][icell]=Res_data.data[3];
		deltT[icell] = fmin(CellArea[icell]*CFL/T, 0.1);
		
	}
/*	else
	{
		dt_s[tid] = 9999999.0; 
	}	
	__syncthreads();
		
	if(blockSize>=512)
	{
		if(tid<256) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+256]);
		__syncthreads();
	}
	
	if(blockSize>=256)
	{
		if(tid<128) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+128]);
		__syncthreads();
	}
	
	if(blockSize>=128)
	{
		if(tid<64) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+64]);
		__syncthreads();
	}
	
	if(blockSize>=64)
	{
		if(tid<32) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+32]);
		__syncthreads();
	}
		
	if(tid<32)
	{
//		if(blockSize >=64 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+32]); 
		if(blockSize >=32 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+16]); 
		if(blockSize >=16 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+8]); 	
		if(blockSize >=8 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+4]); 	
		if(blockSize >=4 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+2]); 	
		if(blockSize >=2 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+1]); 	
	}		
	
	if(tid == 0 ){
		deltT[blockIdx.x] = dt_s[0];	
	}
*/	
}

__global__ void kernel_TimeStep(double* __restrict__ deltT, double* __restrict__ CellArea,
								double CFL, int Ncell)
{
	__shared__ double dt_s[MaxThreads];
	int	blockSize = blockDim.x;
	int icell=blockDim.x * blockIdx.x + threadIdx.x;
	int	tid = threadIdx.x;
	if(icell>0 && icell<=Ncell)
	{	
		dt_s[tid] = CellArea[icell]*CFL/deltT[icell];
	}
	else
	{
		dt_s[tid] = 9999999.0; 
	}	
	__syncthreads();
		
	if(blockSize>=512)
	{
		if(tid<256) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+256]);
		__syncthreads();
	}
	
	if(blockSize>=256)
	{
		if(tid<128) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+128]);
		__syncthreads();
	}
	
	if(blockSize>=128)
	{
		if(tid<64) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+64]);
		__syncthreads();
	}
	
	if(blockSize>=64)
	{
		if(tid<32) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+32]);
		__syncthreads();
	}
		
	if(tid<32)
	{
//		if(blockSize >=64 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+32]); 
		if(blockSize >=32 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+16]); 
		if(blockSize >=16 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+8]); 	
		if(blockSize >=8 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+4]); 	
		if(blockSize >=4 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+2]); 	
		if(blockSize >=2 ) dt_s[tid] = fmin(dt_s[tid],dt_s[tid+1]); 	
	}		
	
	if(tid == 0 ){ deltT[blockIdx.x] = dt_s[0];	
//		cuPrintf("%d %d %f\n",
//	icell, tid,deltT[blockIdx.x]);
	}	

}

__global__ void kernel_RKStep(W_S w_S, W_S w0_S, double RK, 
						double* __restrict__ deltT,
						RESD_S Res_S, double* __restrict__ CellArea, 
						double GAMA, double R, int Ncell)
{
	W   ww;//, ww0;
	int icell=blockDim.x * blockIdx.x + threadIdx.x;
	if(icell>0 && icell<=Ncell)
	{
		RK *= deltT[icell];
//		ww  = w[icell];
//		ww0 =w0[icell];
		
 		ww.density  =w0_S.density[icell]  -RK*Res_S.data[0][icell]/CellArea[icell];
		ww.density_U=w0_S.density_U[icell]-RK*Res_S.data[1][icell]/CellArea[icell];
		ww.density_V=w0_S.density_V[icell]-RK*Res_S.data[2][icell]/CellArea[icell];
		ww.density_E=w0_S.density_E[icell]-RK*Res_S.data[3][icell]/CellArea[icell];		   
				
		kernel_CalculationPTA(ww,GAMA,R);	
		
		w_S.density[icell] = ww.density;
		w_S.density_U[icell] = ww.density_U;
		w_S.density_V[icell] = ww.density_V;
		w_S.density_E[icell] = ww.density_E;
		w_S.P[icell] = ww.P;
		w_S.T[icell] = ww.T;
		w_S.A[icell] = ww.A;
	}	
}								

__global__ void kernel_drou(W_S w_S, W_S w0_S, double* __restrict__ drou, int Ncell)
{
	__shared__ double rou[MaxThreads];
	int icell=blockDim.x * blockIdx.x + threadIdx.x;
	int	blockSize = blockDim.x;
	int	tid = threadIdx.x;	
	if(icell>0 && icell<=Ncell)
	{
		rou[tid]= (w_S.density[icell]-w0_S.density[icell])
				 *(w_S.density[icell]-w0_S.density[icell]);	   		   			   
	}
	else
	{
		rou[tid]=0.0;
	}
	__syncthreads();
	
	if(blockSize>=512)
	{
		if(tid<256) rou[tid] = rou[tid]+rou[tid+256];
		__syncthreads();
	}
	
	if(blockSize>=256)
	{
		if(tid<128) rou[tid] = rou[tid]+rou[tid+128];
		__syncthreads();
	}
	
	if(blockSize>=128)
	{
		if(tid<64) rou[tid] = rou[tid]+rou[tid+64];
		__syncthreads();
	}
	
	if(blockSize>=64)
	{
		if(tid<32) rou[tid] = rou[tid]+rou[tid+32];
		__syncthreads();
	}	
	
	if(tid<32)
	{
//		if(blockSize >=64 ) rou[tid] = rou[tid]+rou[tid+32]; 
		if(blockSize >=32 ) rou[tid] = rou[tid]+rou[tid+16]; 
		if(blockSize >=16 ) rou[tid] = rou[tid]+rou[tid+8]; 	
		if(blockSize >=8 ) rou[tid] = rou[tid]+rou[tid+4]; 	
		if(blockSize >=4 ) rou[tid] = rou[tid]+rou[tid+2]; 	
		if(blockSize >=2 ) rou[tid] = rou[tid]+rou[tid+1]; 	
	}		
	
	if(tid == 0 ) drou[blockIdx.x] = rou[0];			
}


bool cuPrintInit()  
{  
    cudaError_t err = cudaPrintfInit();  
    if(0 != strcmp("no error", cudaGetErrorString(err)))  return false;  
    return true;  
}  


__global__ void displayGPU_demo()  
{  
    int bsize = blockDim.x;  
    int bid = blockIdx.x;  
    int tid = bid * bsize + threadIdx.x;  
    cuPrintf("当前执行kernel的 block 编号:/t%d/n", bid);  
    cuPrintf("当前执行kernel的 thread 在当前块中编号:/t%d/n", threadIdx.x);  
    cuPrintf("当前执行kernel的 thread 全局编号:/t%d/n", tid);  
    cuPrintf("thread over/n/n");  
}  


double Calc_SRI(W &LW,W &RW,EDGE &Iedge)
{
	 double u,v,rou,p,VC,CS,SR,EE;

	   rou=(LW.density+RW.density)/2.0;   //the average value of the left cell & the right cell to an edge
	   u=(LW.density_U+RW.density_U)/rou/2.0;
	   v=(LW.density_V+RW.density_V)/rou/2.0;

	   VC=u*Iedge.vectory+v*Iedge.vectorx;	

	   EE=(LW.density_E+RW.density_E)/2.0/rou-(u*u+v*v)/2.0;
	   p=rou*EE*(GAMA-1.0);  //pressure
	   CS=sqrt(GAMA*p/rou)*sqrt(Iedge.vectorx*Iedge.vectorx+Iedge.vectory*Iedge.vectory);
	   SR=fabs(VC)+CS;
	
	   return(SR);	//variable for timestep computing
}

int Calc_Function(W &w,double &A,double &U,double &V,double &H,double &P)
{
	double E;

	U=w.density_U/w.density;
	V=w.density_V/w.density;
	E=w.density_E/w.density-0.5*(U*U+V*V);

	P=(GAMA-1.0)*w.density*E;
	A=sqrt(GAMA*P/w.density);
	H=w.density_E/w.density+P/w.density;

	return 0;
}

int Far_Boundary(W *w,W &w0,W &DW,EDGE &Iedge)
{
	double PB,PD,PA,nx,ny,nn,Un,Uncell,AD,ROUD,UB,VB,UD,VD;   //B-boundary,D-domain,A-Initial variable

	nn=sqrt(Iedge.vectorx*Iedge.vectorx+Iedge.vectory*Iedge.vectory);
	nx=Iedge.vectory/nn;   //gradient
	ny=Iedge.vectorx/nn;
	
	UD=DW.density_U/DW.density;
	VD=DW.density_V/DW.density;

	Un=fabs(UI*nx+VI*ny);		//contravariant velocity of the coming flow
	Uncell=UD*nx+VD*ny;			//contravariant velocity of the cell's flow

	if(Un/AIN>1.0)   //supersonic
	{
		if(Uncell>0.0)			//outflow
			w0=DW;

		if(Uncell<=0.0)			//infow
		{
			w0.density  =1.0;
			w0.density_U=UI;
			w0.density_V=VI;
			w0.density_E=R*TIN/(VIN*VIN)/(GAMA-1.0)+0.50;

			CalculationPTA(w0);	
		}		
	}

	else		       //subsonic
	{
		PA	=PIS;
		PD	=DW.P;
		AD	=DW.A;
		ROUD=DW.density;

		if(Uncell>0.0)//outflow
		{
			PB=PA;
			w0.P=PB;
			w0.density=ROUD+(PB-PD)/AD/AD;

			UB=UD+nx*(PD-PB)/ROUD/AD;
			VB=VD+ny*(PD-PB)/ROUD/AD;

			w0.density_U=UB*w0.density;
			w0.density_V=VB*w0.density;
			w0.density_E=PB/(GAMA-1.0)+0.5*(UB*UB+VB*VB)*w0.density;

			CalculationPTA(w0);	
		}

		if(Uncell<=0.0)//inflow
		{
			PB=0.5*( PA+PD- ROUD*AD* ( nx*(UI-UD)+ny*(VI-VD) ) );
			w0.P=PB;
			
			w0.density=1.0+(PB-PA)/AD/AD;
			
			UB=UI-nx*(PA-PB)/ROUD/AD;
			VB=VI-ny*(PA-PB)/ROUD/AD;

			w0.density_U=UB*w0.density;
			w0.density_V=VB*w0.density;
			w0.density_E=PB/(GAMA-1.0)+0.5*(UB*UB+VB*VB)*w0.density;

			CalculationPTA(w0);	
		}
	}

	return 0;
}
 
int  Boundary(W *w,W *Wallw,W *Farw,CELL *cell,EDGE *edge){
	int iedge,wallid,farid;
	double UU,VV,UV;
	double nij[2];

	for(iedge=1;iedge<=Nedge;iedge++){

		if(edge[iedge].ELog==1){  //FarFeild

			farid=edge[iedge].farfieldid;
			Far_Boundary(w,Farw[farid], w[edge[iedge].left_cell] , edge[iedge]);
		}
			
		if(edge[iedge].ELog==2){  //Wall
		
			nij[0]=edge[iedge].vectory/edge[iedge].vectorn;
			nij[1]=edge[iedge].vectorx/edge[iedge].vectorn;

			UU=w[edge[iedge].left_cell].density_U/w[edge[iedge].left_cell].density;
			VV=w[edge[iedge].left_cell].density_V/w[edge[iedge].left_cell].density;
			UV=UU*nij[0]+VV*nij[1];

			UU=UU-2*UV*nij[0];
			VV=VV-2*UV*nij[1];
		
			wallid=edge[iedge].wallid;
			Wallw[wallid]=w[edge[iedge].left_cell];
			Wallw[wallid].density_U=Wallw[wallid].density*UU;
			Wallw[wallid].density_V=Wallw[wallid].density*VV;
//			printf("%d %d %d %f %f\n",iedge, wallid, edge[iedge].left_cell,Wallw[wallid].density_U,Wallw[wallid].density_V);
		}

	}

	return 0;
}

int Gradient(W *w,W *Wallw,W *Farw,CELL *cell,EDGE *edge,double *CellArea,NODE *node){
		int    i,j,k,neigh,icell;
		double nij[2],ds;
		double Ujmax[4],Ujmin[4],comp[4],vectorx,vectory;
		int	   IP[3]={1,2,0};
		W ghost;

		for(icell=1;icell<=Ncell;icell++){

			for(j=0;j<2;j++)
				for(k=0;k<4;k++)
					cell[icell].deltU[j][k]=0.0;

			for(i=0;i<3;i++){	

				int ie1=IP[i];
				int ie2=IP[ie1];
				int N1=cell[icell].Point[ie1];
				int N2=cell[icell].Point[ie2];

				vectorx=node[N1].x-node[N2].x;    //2->3,3->1,1->2
				vectory=node[N2].y-node[N1].y;

				for(j=0;j<3;j++){					//find the neighbor cell connecting edge
					int NN1=edge[cell[icell].neighbor[j].celledge].node1;
					int NN2=edge[cell[icell].neighbor[j].celledge].node2;

					if( (NN1==N1&&NN2==N2) || (NN1==N2&&NN2==N1) )
					{
						break;
					}
				}

				neigh =cell[icell].neighbor[j].neicell;					    //the i-th neighbor's cell number
				ds    =edge[cell[icell].neighbor[j].celledge].vectorn;	//edge with the i-th neighbor's cell 
				nij[0]=vectory/ds;
				nij[1]=vectorx/ds;

				 if(neigh >0){   //domain
						comp[0]=w[neigh].density;
						comp[1]=w[neigh].density_U;
						comp[2]=w[neigh].density_V;
						comp[3]=w[neigh].density_E;
					}
				
				else{
						if(neigh==-1) // farfield
								ghost=Farw[edge[cell[icell].neighbor[j].celledge].farfieldid];
								
						if(neigh==-2) //wall
								ghost=Wallw[edge[cell[icell].neighbor[j].celledge].wallid];
							
						comp[0]=ghost.density;
						comp[1]=ghost.density_U;
						comp[2]=ghost.density_V;
						comp[3]=ghost.density_E;	
				}
					
					if(i==0)     													//search the Ujmin/max
						for(j=0;j<4;j++){
							Ujmax[j]=comp[j];
							Ujmin[j]=comp[j];
						}

					else
						for(j=0;j<4;j++){
							if(comp[j]<Ujmin[j])
								Ujmin[j]=comp[j];

							if(comp[j]>Ujmax[j])
								Ujmax[j]=comp[j];
						}
/*						
if(i==1 && icell==684)
		printf("%d, %E, %E, %E, %E, %E, %E, %E, %E\n",icell,
			cell[icell].deltU[0][0],cell[icell].deltU[0][1],cell[icell].deltU[0][2],cell[icell].deltU[0][3],
			cell[icell].deltU[1][0],cell[icell].deltU[1][1],cell[icell].deltU[1][2],cell[icell].deltU[1][3]);				
if(i==1 && icell==684)
		printf("%d, %E, %E, %E, %E, %E, %E, %E, %E\n",icell,
			w[icell].density,w[icell].density_U,w[icell].density_V,w[icell].density_E,
			comp[0],comp[1],comp[2],comp[3]);	
if(i==1 && icell==684)
		printf("%d, %E, %E, %E\n", icell,
				nij[0],nij[1],ds);			
*/				
					for(j=0;j<2;j++){											//j=0 for x direct j=1 for y direct 
						cell[icell].deltU[j][0]+=0.5*(w[icell].density   + comp[0] )*nij[j]*ds;
						cell[icell].deltU[j][1]+=0.5*(w[icell].density_U + comp[1] )*nij[j]*ds;
						cell[icell].deltU[j][2]+=0.5*(w[icell].density_V + comp[2] )*nij[j]*ds;
						cell[icell].deltU[j][3]+=0.5*(w[icell].density_E + comp[3] )*nij[j]*ds;
					}

				
				//3 neighbors search finished
			}

//printf("%d, %f\n", icell, 	CellArea[icell]);
			for(j=0;j<2;j++)
				for(k=0;k<4;k++)
					cell[icell].deltU[j][k] /= CellArea[icell];
/*
		printf("%d, %E, %E, %E, %E, %E, %E, %E, %E\n",icell,
			cell[icell].deltU[0][0],cell[icell].deltU[0][1],cell[icell].deltU[0][2],cell[icell].deltU[0][3],
			cell[icell].deltU[1][0],cell[icell].deltU[1][1],cell[icell].deltU[1][2],cell[icell].deltU[1][3]);				
*/	
			comp[0]=w[icell].density;		//Ui
			comp[1]=w[icell].density_U;
			comp[2]=w[icell].density_V;
			comp[3]=w[icell].density_E;

			for(j=0;j<4;j++){							//search the Umin/max
				if(comp[j]>Ujmax[j])
					cell[icell].Umax[j]=comp[j];
				else
					cell[icell].Umax[j]=Ujmax[j];

				if(comp[j]<Ujmin[j])
					cell[icell].Umin[j]=comp[j];
				else
					cell[icell].Umin[j]=Ujmin[j];
				}

		}

		return 0;
}


int Limiter(W *w,EDGE &Iedge,int Cellnum,CELL *cell,EDGE *edge,double fi[4],double delt2[4]){
		double UI[4];
		int i;

//		r[0]=Iedge.midx-cell[Iedge.left_cell].center[0];
//		r[1]=Iedge.midy-cell[Iedge.left_cell].center[1];

		UI[0]=w[Cellnum].density;
		UI[1]=w[Cellnum].density_U;
		UI[2]=w[Cellnum].density_V;
		UI[3]=w[Cellnum].density_E;

		for(i=0;i<4;i++){
			if(delt2[i] > 0 )
				fi[i] = 1.0 > (( cell[Cellnum].Umax[i]-UI[i] )/delt2[i]) ? ( cell[Cellnum].Umax[i]-UI[i] )/delt2[i] : 1.0;

			if(delt2[i] <0 )
				fi[i] = 1.0 > (( cell[Cellnum].Umin[i]-UI[i] )/delt2[i]) ? ( cell[Cellnum].Umin[i]-UI[i] )/delt2[i] : 1.0;

			if(delt2[i] ==0)
				fi[i] = 1.0 ;
		}

		return 0;
}

int LBJinterpolate(W *w,EDGE &Iedge,CELL *cell,EDGE *edge,W &LW,W &RW,double *CellArea,NODE *node){
		W UI,UJ;
		double fiI[4],fiJ[4],rL[2],rR[2],delt2I[4],delt2J[4];
		int i;
		
		UI=w[Iedge.left_cell];
		UJ=w[Iedge.right_cell];

		rL[0]=Iedge.midx-cell[Iedge.left_cell].center[0];
		rL[1]=Iedge.midy-cell[Iedge.left_cell].center[1];

		rR[0]=Iedge.midx-cell[Iedge.right_cell].center[0];
		rR[1]=Iedge.midy-cell[Iedge.right_cell].center[1];

		for(i=0;i<4;i++){
			delt2I[i]=cell[Iedge.left_cell ].deltU[0][i]*rL[0]+cell[Iedge.left_cell ].deltU[1][i]*rL[1];
			delt2J[i]=cell[Iedge.right_cell].deltU[0][i]*rR[0]+cell[Iedge.right_cell].deltU[1][i]*rR[1];
		}

		Limiter(w,Iedge,Iedge.left_cell ,cell,edge,fiI,delt2I);
		Limiter(w,Iedge,Iedge.right_cell,cell,edge,fiJ,delt2J);

		LW.density  =UI.density  +fiI[0]*delt2I[0];
		LW.density_U=UI.density_U+fiI[1]*delt2I[1];
		LW.density_V=UI.density_V+fiI[2]*delt2I[2];
		LW.density_E=UI.density_E+fiI[3]*delt2I[3];

		RW.density  =UJ.density  +fiJ[0]*delt2J[0];
		RW.density_U=UJ.density_U+fiJ[1]*delt2J[1];
		RW.density_V=UJ.density_V+fiJ[2]*delt2J[2];
		RW.density_E=UJ.density_E+fiJ[3]*delt2J[3];

		return 0;
}

int ROE_SCHEME(W &LW,W &RW,double nx,double ny,double FLUX[4]){

	double ROUB,UB,VB,HB,QB,VC,AB,DVC,epsilon;
	double A[2],U[2],V[2],H[2],P[2],M[2],FF[3][4],AD[3],WL[4],WR[4];
	double lrou,rrou;

	Calc_Function(LW,A[0],U[0],V[0],H[0],P[0]);
	Calc_Function(RW,A[1],U[1],V[1],H[1],P[1]);

	M[0]=(U[0]*nx+V[0]*ny);	
	M[1]=(U[1]*nx+V[1]*ny);	

	DVC=M[1]-M[0];

	lrou=LW.density;
	rrou=RW.density;
    
	ROUB=sqrt ( lrou * rrou );

	UB=( U[0] * sqrt( lrou )+U[1] * sqrt( rrou ) )
		/(sqrt( lrou )+sqrt( rrou ));

	VB=( V[0] * sqrt( lrou )+V[1] * sqrt( rrou ) )
		/(sqrt( lrou )+sqrt( rrou ));

	HB=( H[0] * sqrt( lrou )+H[1] * sqrt( rrou ) )
		/(sqrt( lrou )+sqrt( rrou ));

	QB=UB*UB+VB*VB;
	AB=sqrt( (GAMA-1.0) * (HB-QB/2.0) );
	VC=UB*nx+VB*ny;  
	
	epsilon=0.05;

	if(fabs(VC-AB)<epsilon)
		AD[0]=((VC-AB)*(VC-AB)+epsilon*epsilon)/(2*epsilon);
	else
		AD[0]=fabs(VC-AB);

	if(fabs(VC)<epsilon)
		AD[1]=(VC*VC+epsilon*epsilon)/(2*epsilon);
	else
		AD[1]=fabs(VC);

	if(fabs(VC+AB)<epsilon)
		AD[2]=((VC+AB)*(VC+AB)+epsilon*epsilon)/(2*epsilon);
	else
		AD[2]=fabs(VC+AB);

	FF[0][0]=AD[0]*((P[1]-P[0])-ROUB*AB*DVC)/(2.0*AB*AB);
	FF[0][1]=FF[0][0]*(UB-AB*nx);
	FF[0][2]=FF[0][0]*(VB-AB*ny);
	FF[0][3]=FF[0][0]*(HB-AB*VC);
	
	FF[1][0]=AD[1]*((rrou-lrou)-(P[1]-P[0])/(AB*AB));
	FF[1][1]=FF[1][0]*UB+AD[1]*ROUB*(U[1]-U[0]-DVC*nx);
	FF[1][2]=FF[1][0]*VB+AD[1]*ROUB*(V[1]-V[0]-DVC*ny);
	FF[1][3]=FF[1][0]*QB/2.0+AD[1]*ROUB*(UB*(U[1]-U[0])+VB*(V[1]-V[0])-DVC*VC);
	
	FF[2][0]=AD[2]*((P[1]-P[0])+ROUB*AB*DVC)/(2.0*AB*AB);
	FF[2][1]=FF[2][0]*(UB+AB*nx);
	FF[2][2]=FF[2][0]*(VB+AB*ny);
	FF[2][3]=FF[2][0]*(HB+AB*VC);

	WL[0]=lrou*M[0];
	WL[1]=U[0]*WL[0]+nx*P[0];
	WL[2]=V[0]*WL[0]+ny*P[0];
	WL[3]=WL[0]*H[0];
	
	WR[0]=rrou*M[1];
	WR[1]=U[1]*WR[0]+nx*P[1];
	WR[2]=V[1]*WR[0]+ny*P[1];
	WR[3]=WR[0]*H[1];

	for(int k=0;k<4;k++)		//ROE scheme
		FLUX[k]=0.5*((WL[k]+WR[k])-(FF[0][k]+FF[1][k]+FF[2][k]));

	return 0;
}

int Flux(W *w,W *Wallw,W *Farw,EDGE &Iedge,double FLUX[4],double *CellArea,double *deltT,int iedge,CELL *cell,EDGE *edge,NODE *node)  //for the edge
{
	double nn,nx,ny;
	W wi,LW,RW;
	
	nn=sqrt(Iedge.vectorx*Iedge.vectorx+Iedge.vectory*Iedge.vectory);
	nx=Iedge.vectory/nn;   //gradient of the edge
	ny=Iedge.vectorx/nn;

	if(Iedge.ELog==0)//inside the fluid field
	{
		LBJinterpolate(w,Iedge,cell,edge,LW,RW,CellArea,node);
		ROE_SCHEME(LW,RW,nx,ny,FLUX);

		deltT[Iedge.left_cell]+=Calc_SRI(w[Iedge.left_cell],w[Iedge.right_cell],Iedge); 

		if(Iedge.right_cell!=-1)
			deltT[Iedge.right_cell]+=Calc_SRI(w[Iedge.left_cell],w[Iedge.right_cell],Iedge);
		
	}

    if(Iedge.ELog==1)   //FarField
    {
		wi=*(Farw+Iedge.farfieldid);
		deltT[Iedge.left_cell]+=Calc_SRI(w[Iedge.left_cell],wi,Iedge); 

		ROE_SCHEME(w[Iedge.left_cell],wi,nx,ny,FLUX);
    }

	if(Iedge.ELog==2)  //Wall
	{
		
		wi=*(Wallw+Iedge.wallid);
		deltT[Iedge.left_cell]+=Calc_SRI(w[Iedge.left_cell],wi,Iedge); 

		ROE_SCHEME(w[Iedge.left_cell],wi,nx,ny,FLUX);
		
		/*
		FLUX[0]=0;
		FLUX[1]=(w+Iedge->left_cell)->P*nx;
		FLUX[2]=(w+Iedge->left_cell)->P*ny;
		FLUX[3]=0;

		deltT[Iedge->left_cell]+=sqrt(GAMA*(w+Iedge->left_cell)->P/(w+Iedge->left_cell)->density)*nn;*/
    }

	return 0;
}

void copyH2D()
{

	cudaMemcpy(d_cell, cell, (Ncell+1)*sizeof(CELL), cudaMemcpyHostToDevice);
	cudaMemcpy(d_node, node, (Nnode+1)*sizeof(NODE), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_node_S.x, node_S.x, (Nnode+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_node_S.y, node_S.y, (Nnode+1)*sizeof(double), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_cell_S.CLog, cell_S.CLog, (Ncell+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cell_S.Point[0], cell_S.Point[0], (Ncell+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cell_S.Point[1], cell_S.Point[1], (Ncell+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cell_S.Point[2], cell_S.Point[2], (Ncell+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cell_S.center[0], cell_S.center[0], (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cell_S.center[0], cell_S.center[1], (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cell_S.celledge[0], cell_S.celledge[0], (Ncell+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cell_S.celledge[1], cell_S.celledge[1], (Ncell+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cell_S.celledge[2], cell_S.celledge[2], (Ncell+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cell_S.neicell[0], cell_S.neicell[0], (Ncell+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cell_S.neicell[1], cell_S.neicell[1], (Ncell+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_cell_S.neicell[2], cell_S.neicell[2], (Ncell+1)*sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_CellArea, CellArea, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(d_deltT, deltT, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_w_S.density	, w0_S.density	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w_S.density_U	, w0_S.density_U	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w_S.density_V	, w0_S.density_V	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w_S.density_E	, w0_S.density_E	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w_S.P	, w0_S.P	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w_S.T	, w0_S.T	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w_S.A	, w0_S.A	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_w0_S.density	, w0_S.density	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w0_S.density_U	, w0_S.density_U	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w0_S.density_V	, w0_S.density_V	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w0_S.density_E	, w0_S.density_E	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w0_S.P	, w0_S.P	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w0_S.T	, w0_S.T	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w0_S.A	, w0_S.A	, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	

	cudaMemcpy(d_Res_S.data[0], Res_S.data[0], (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Res_S.data[1], Res_S.data[1], (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Res_S.data[2], Res_S.data[2], (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Res_S.data[3], Res_S.data[3], (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);

	cudaMemcpy(d_edge_S.ELog, edge_S.ELog, (Nedge+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_S.left_cell, edge_S.left_cell, (Nedge+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_S.right_cell, edge_S.right_cell, (Nedge+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_S.farfieldid, edge_S.farfieldid, (Nedge+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_S.wallid, edge_S.wallid, (Nedge+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_S.node1, edge_S.node1, (Nedge+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_S.node2, edge_S.node2, (Nedge+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_S.vectorn, edge_S.vectorn, (Nedge+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_S.vectorx, edge_S.vectorx, (Nedge+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_S.vectory, edge_S.vectory, (Nedge+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_S.midx, edge_S.midx, (Nedge+1)*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge_S.midy, edge_S.midy, (Nedge+1)*sizeof(double), cudaMemcpyHostToDevice);
	
}

void copyD2H()
{

	cudaMemcpy(cell, d_cell, (Ncell+1)*sizeof(CELL), cudaMemcpyDeviceToHost);
	cudaMemcpy(node, d_node, (Nnode+1)*sizeof(NODE), cudaMemcpyDeviceToHost);
	cudaMemcpy(CellArea, d_CellArea, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);

	cudaMemcpy(deltT, d_deltT, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);

	cudaMemcpy(w0_S.density	, d_w0_S.density	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w0_S.density_U	, d_w0_S.density_U	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w0_S.density_V	, d_w0_S.density_V	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w0_S.density_E	, d_w0_S.density_E	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w0_S.P	, d_w0_S.P	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w0_S.T	, d_w0_S.T	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w0_S.A	, d_w0_S.A	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaMemcpy(w_S.density	, d_w_S.density	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w_S.density_U	, d_w_S.density_U	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w_S.density_V	, d_w_S.density_V	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w_S.density_E	, d_w_S.density_E	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w_S.P	, d_w_S.P	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w_S.T	, d_w_S.T	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(w_S.A	, d_w_S.A	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);			

	cudaMemcpy(Res_S.data[0], d_Res_S.data[0], (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Res_S.data[1], d_Res_S.data[1], (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Res_S.data[2], d_Res_S.data[2], (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Res_S.data[3], d_Res_S.data[3], (Ncell+1)*sizeof(double), cudaMemcpyDeviceToHost);


	cudaMemcpy(edge_S.ELog, d_edge_S.ELog, (Nedge+1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_S.left_cell, d_edge_S.left_cell, (Nedge+1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_S.right_cell, d_edge_S.right_cell, (Nedge+1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_S.farfieldid, d_edge_S.farfieldid, (Nedge+1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_S.wallid, d_edge_S.wallid, (Nedge+1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_S.node1, d_edge_S.node1, (Nedge+1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_S.node2, d_edge_S.node2, (Nedge+1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_S.vectorn, d_edge_S.vectorn, (Nedge+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_S.vectorx, d_edge_S.vectorx, (Nedge+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_S.vectory, d_edge_S.vectory, (Nedge+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_S.midx, d_edge_S.midx, (Nedge+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge_S.midy, d_edge_S.midy, (Nedge+1)*sizeof(double), cudaMemcpyDeviceToHost);
	

	cudaMemcpy(Wallw_S.density	, d_Wallw_S.density	, (WallBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Wallw_S.density_U	, d_Wallw_S.density_U	, (WallBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Wallw_S.density_V	, d_Wallw_S.density_V	, (WallBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Wallw_S.density_E	, d_Wallw_S.density_E	, (WallBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Wallw_S.P	, d_Wallw_S.P	, (WallBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Wallw_S.T	, d_Wallw_S.T	, (WallBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Wallw_S.A	, d_Wallw_S.A	, (WallBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);	

	cudaMemcpy(Farw_S.density	, d_Farw_S.density	, (FarBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Farw_S.density_U	, d_Farw_S.density_U	, (FarBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Farw_S.density_V	, d_Farw_S.density_V	, (FarBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Farw_S.density_E	, d_Farw_S.density_E	, (FarBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Farw_S.P	, d_Farw_S.P	, (FarBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Farw_S.T	, d_Farw_S.T	, (FarBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Farw_S.A	, d_Farw_S.A	, (FarBoundNum+1)*sizeof(double), cudaMemcpyDeviceToHost);	

		
}


#define eps 0.8

__global__ void kernel_Smooth1(RESD_S Res_S,
								RESD_S R1_S,
								CELL* __restrict__ cell,
								int Ncell)
{
	int i;
	double data[4];
	int neicell[3];
		
	int icell=blockDim.x * blockIdx.x + threadIdx.x;
	
	if(icell>0 && icell<=Ncell)
	{
		neicell[0] = cell[icell].neighbor[0].neicell;
		neicell[1] = cell[icell].neighbor[1].neicell;
		neicell[2] = cell[icell].neighbor[2].neicell;
		
		for(i=0;i<4;i++)
		{
			data[i]=0.0;
		}
		
		for(i=0;i<4;i++)
		{
			if(neicell[0] > 0)
				data[i] += Res_S.data[i][neicell[0]];
			if(neicell[1] > 0)
				data[i] += Res_S.data[i][neicell[1]];
			if(neicell[2] > 0)
				data[i] += Res_S.data[i][neicell[2]];
		}
		
		for(i=0;i<4;i++)
		{
			R1_S.data[i][icell] = ( data[i] * (eps) + Res_S.data[i][icell] )
							/( 1.0 + (eps) * cell[icell].CLog );
		}
	}
}

__global__ void kernel_Smooth2(RESD_S Res_S,
								RESD_S R1_S,
								CELL* __restrict__ cell,
								int Ncell)
{
	int i;
	double data[4];
	double data2[4];
	int neicell[3];
		
	int icell=blockDim.x * blockIdx.x + threadIdx.x;
	
	if(icell>0 && icell<=Ncell)
	{
		neicell[0] = cell[icell].neighbor[0].neicell;
		neicell[1] = cell[icell].neighbor[1].neicell;
		neicell[2] = cell[icell].neighbor[2].neicell;
		
		for(i=0;i<4;i++)
		{
			data[i]=0.0;
		}
		
		for(i=0;i<4;i++)
		{
			if(neicell[0] > 0)
				data[i] += R1_S.data[i][neicell[0]];
			if(neicell[1] > 0)
				data[i] += R1_S.data[i][neicell[1]];
			if(neicell[2] > 0)
				data[i] += R1_S.data[i][neicell[2]];
		}
		
		for(i=0;i<4;i++)
		{
			data2[i] = ( data[i] * (eps) + Res_S.data[i][icell])
					/ ( 1.0 + (eps) * cell[icell].CLog );
		}
		
		for(i=0;i<4;i++)
		{
			if( fabs(Res_S.data[i][icell]) > fabs(data2[i]))
				Res_S.data[i][icell] = data2[i];
		}		
	}
}

void Smooth(int blockNum, int threadNum )
{
	kernel_Smooth1<<< blockNum, threadNum>>>
					(d_Res_S, d_R1_S, d_cell, Ncell);
	kernel_Smooth2<<< blockNum, threadNum>>>
					(d_Res_S, d_R1_S, d_cell, Ncell);
}

void changeCell() // 此函数有莫名错误，非常危险。。。。
{
	EDGE	*newEdge =(EDGE *)calloc(Nedge+1,sizeof(EDGE));
	CELL	*newCell =(CELL *)calloc(Ncell+1,sizeof(CELL));
	int		*cellNewInOld = (int *) calloc( Ncell+1,sizeof(int));
	int		*cellOldInNew = (int *) calloc( Ncell+1,sizeof(int));
	int i=1;
	int j=1;
	int iedge;
	
	
	for(i=1; i<Nedge+1; i++)
	{
		newEdge[i] = edge[i];
	}
	
	for(i=1; i<Ncell+1; i++)
	{
		cellNewInOld[i]=0;
		cellOldInNew[i]=0;
	}
	
	for( i=1; i<Ncell+1; i++)
	{
		if( cell[i].neighbor[0].neicell >0  &&
			cell[i].neighbor[1].neicell >0 &&
			cell[i].neighbor[2].neicell >0 )
		{
			newCell[j] = cell[i];
			cellNewInOld[i]=j;
			cellOldInNew[j]=i;
			j++;
		}
	}	

	for( i=1; i<Ncell+1; i++)
	{
		if( cell[i].neighbor[0].neicell==-1 ||
			cell[i].neighbor[1].neicell==-1 ||
			cell[i].neighbor[2].neicell==-1 )
		{
			newCell[j] = cell[i];
			cellNewInOld[i]=j;
			cellOldInNew[j]=i;
			j++;
		}
	}

	for( i=1; i<Ncell+1; i++)
	{
		if( cell[i].neighbor[0].neicell==-2 ||
			cell[i].neighbor[1].neicell==-2 ||
			cell[i].neighbor[2].neicell==-2 )
		{
			newCell[j] = cell[i];
			cellNewInOld[i]=j;
			cellOldInNew[j]=i;
			j++;
		}
	}
	
	for(i=1; i<Nedge+1; i++)
	{
		int icell = edge[i].left_cell;
		if(icell > 0)
			newEdge[i].left_cell = cellNewInOld[icell];
			
		icell = edge[i].right_cell;
		if(icell > 0)
			newEdge[i].right_cell = cellNewInOld[icell];
	}

	for( i=1; i<Ncell+1; i++)
	{
		for(int m=0; m<3; m++)
		{
			int p = cellOldInNew[i];
			newCell[i].Point[m] = cell[p].Point[m];
			j = newCell[i].neighbor[m].neicell;
			if(j>0)
			{
				newCell[i].neighbor[m].neicell = cellNewInOld[j];
			}
		}
	}

	for(i=1; i<Ncell+1; i++)
	{
		cell[i]=newCell[i];
	}
	
	for(i=1; i<Nedge+1; i++)
	{
		edge[i]=newEdge[i];
	}

	free(newCell);
	free(newEdge);
	free(cellNewInOld);
	free(cellOldInNew);
}

void changeEdge()
{
	EDGE	*newEdge =(EDGE *)calloc(Nedge+1,sizeof(EDGE));
	CELL	*newCell =(CELL *)calloc(Ncell+1,sizeof(CELL));
	int		*edgeNewInOld = (int *) calloc( Nedge+1,sizeof(int));
	int i=1;
	int j=1;
	int icell;
	CELL	Icell;
	
	for(i=1; i<Ncell+1; i++)
	{
		newCell[i]=cell[i];
	}
	
	for(i=1; i<Nedge+1; i++)
	{
		if(edge[i].ELog == 0)
		{
			newEdge[j]=edge[i];
			edgeNewInOld[i] = j;
			j++;
		}
	}

	for(i=1; i<Nedge+1; i++)
	{
		if(edge[i].ELog == 1)
		{
			newEdge[j]=edge[i];
			edgeNewInOld[i] = j;
			j++;
		}
	}
	
	for(i=1; i<Nedge+1; i++)
	{
		if(edge[i].ELog == 2)
		{
			newEdge[j]=edge[i];
			edgeNewInOld[i] = j;
			j++;
		}
	}

	for(i=1; i<Nedge+1; i++)
	{
		for(int m=0; m<3; m++)
		{
			icell=newEdge[i].left_cell;
			j=cell[icell].neighbor[m].celledge; 
			newCell[icell].neighbor[m].celledge = edgeNewInOld[j];
			if(newEdge[i].right_cell>0)
			{
				icell=newEdge[i].right_cell;
				j=cell[icell].neighbor[m].celledge; 
				newCell[icell].neighbor[m].celledge = edgeNewInOld[j];
			}
		}
	}

	for(i=1; i<Ncell+1; i++)
	{
		cell[i]=newCell[i];
	}
	
	for(i=1; i<Nedge+1; i++)
	{
		edge[i]=newEdge[i];
	}

	free(newCell);
	free(newEdge);
	free(edgeNewInOld);
}

void cpuMalloc(int threadNum)
{
	deltT   =(double*)calloc(Ncell+1,sizeof(double));

	node_S.x = (double  *)calloc(Nnode+1,sizeof(double  ));
	node_S.y = (double  *)calloc(Nnode+1,sizeof(double  ));
	
	cell_S.CLog = (int  *)calloc(Ncell+1,sizeof(int  ));
	cell_S.Point[0] = (int  *)calloc(Ncell+1,sizeof(int  ));
	cell_S.Point[1] = (int  *)calloc(Ncell+1,sizeof(int  ));
	cell_S.Point[2] = (int  *)calloc(Ncell+1,sizeof(int  ));
	cell_S.deltU[0][0] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.deltU[0][1] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.deltU[0][2] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.deltU[0][3] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.deltU[1][0] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.deltU[1][1] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.deltU[1][2] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.deltU[1][3] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.Umax[0] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.Umax[1] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.Umax[2] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.Umax[3] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.Umin[0] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.Umin[1] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.Umin[2] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.Umin[3] = (double  *)calloc(Ncell+1,sizeof(double  ));	
	cell_S.center[0] = (double  *)calloc(Ncell+1,sizeof(double  ));
	cell_S.center[1] = (double  *)calloc(Ncell+1,sizeof(double  ));	
	cell_S.celledge[0] = (int  *)calloc(Ncell+1,sizeof(int  ));
	cell_S.celledge[1] = (int  *)calloc(Ncell+1,sizeof(int  ));
	cell_S.celledge[2] = (int  *)calloc(Ncell+1,sizeof(int  ));
	cell_S.neicell[0] = (int  *)calloc(Ncell+1,sizeof(int  ));
	cell_S.neicell[1] = (int  *)calloc(Ncell+1,sizeof(int  ));
	cell_S.neicell[2] = (int  *)calloc(Ncell+1,sizeof(int  ));

	Res_S.data[0] =(double  *)calloc(Ncell+1,sizeof(double  ));
	Res_S.data[1] =(double  *)calloc(Ncell+1,sizeof(double  ));
	Res_S.data[2] =(double  *)calloc(Ncell+1,sizeof(double  ));
	Res_S.data[3] =(double  *)calloc(Ncell+1,sizeof(double  ));
	
	edge_S.ELog = (int  *)calloc(Nedge+1,sizeof(int  ));
	edge_S.left_cell = (int  *)calloc(Nedge+1,sizeof(int  ));
	edge_S.right_cell = (int  *)calloc(Nedge+1,sizeof(int  ));
	edge_S.vectory = (double  *)calloc(Nedge+1,sizeof(double  ));
	edge_S.midx = (double  *)calloc(Nedge+1,sizeof(double  ));
	edge_S.midy = (double  *)calloc(Nedge+1,sizeof(double  ));
	
//	w0      =(W     *)calloc(Ncell+1,sizeof(W     ));	
	
	w0_S.density     =(double     *)calloc(Ncell+1,sizeof(double));
	w0_S.density_U     =(double     *)calloc(Ncell+1,sizeof(double));
	w0_S.density_V     =(double     *)calloc(Ncell+1,sizeof(double));
	w0_S.density_E     =(double     *)calloc(Ncell+1,sizeof(double));
	w0_S.P     =(double     *)calloc(Ncell+1,sizeof(double));
	w0_S.T     =(double     *)calloc(Ncell+1,sizeof(double));
	w0_S.A     =(double     *)calloc(Ncell+1,sizeof(double));
	
	w_S.density     =(double     *)calloc(Ncell+1,sizeof(double));
	w_S.density_U     =(double     *)calloc(Ncell+1,sizeof(double));
	w_S.density_V     =(double     *)calloc(Ncell+1,sizeof(double));
	w_S.density_E     =(double     *)calloc(Ncell+1,sizeof(double));
	w_S.P     =(double     *)calloc(Ncell+1,sizeof(double));
	w_S.T     =(double     *)calloc(Ncell+1,sizeof(double));
	w_S.A     =(double     *)calloc(Ncell+1,sizeof(double));	
	
	Wallw_S.density     =(double     *)calloc(WallBoundNum+1,sizeof(double));
	Wallw_S.density_U     =(double     *)calloc(WallBoundNum+1,sizeof(double));
	Wallw_S.density_V     =(double     *)calloc(WallBoundNum+1,sizeof(double));
	Wallw_S.density_E     =(double     *)calloc(WallBoundNum+1,sizeof(double));
	Wallw_S.P     =(double     *)calloc(WallBoundNum+1,sizeof(double));
	Wallw_S.T     =(double     *)calloc(WallBoundNum+1,sizeof(double));
	Wallw_S.A     =(double     *)calloc(WallBoundNum+1,sizeof(double));
	
	Farw_S.density     =(double     *)calloc(FarBoundNum+1,sizeof(double));
	Farw_S.density_U     =(double     *)calloc(FarBoundNum+1,sizeof(double));
	Farw_S.density_V     =(double     *)calloc(FarBoundNum+1,sizeof(double));
	Farw_S.density_E     =(double     *)calloc(FarBoundNum+1,sizeof(double));
	Farw_S.P     =(double     *)calloc(FarBoundNum+1,sizeof(double));
	Farw_S.T     =(double     *)calloc(FarBoundNum+1,sizeof(double));
	Farw_S.A     =(double     *)calloc(FarBoundNum+1,sizeof(double));
	edge_S.farfieldid = (int  *)calloc(Nedge+1,sizeof(int  ));
	edge_S.wallid = (int  *)calloc(Nedge+1,sizeof(int  ));
	edge_S.node1 = (int  *)calloc(Nedge+1,sizeof(int  ));
	edge_S.node2 = (int  *)calloc(Nedge+1,sizeof(int  ));
	edge_S.vectorn = (double  *)calloc(Nedge+1,sizeof(double  ));
	edge_S.vectorx = (double  *)calloc(Nedge+1,sizeof(double  ));
	edge_S.vectory = (double  *)calloc(Nedge+1,sizeof(double  ));
	edge_S.midx = (double  *)calloc(Nedge+1,sizeof(double  ));
	edge_S.midy = (double  *)calloc(Nedge+1,sizeof(double  ));
//	Wallw   =(W    *)calloc(WallBoundNum+1,sizeof(W));
//	Farw    =(W    *)calloc(FarBoundNum+1 ,sizeof(W));
	
	
	rou     =(double*)calloc(Ncell/threadNum + 1 ,sizeof(double));   
}

void gpuMalloc(int threadNum)
{
	
	cudaMalloc((void**)&d_deltT_e_S.left, (Nedge+1)*sizeof(double));
	cudaMalloc((void**)&d_deltT_e_S.right, (Nedge+1)*sizeof(double));
	
	cudaMalloc((void**)&d_node_S.x, (Nnode+1)*sizeof(double));
	cudaMalloc((void**)&d_node_S.y, (Nnode+1)*sizeof(double));

	cudaMalloc((void**)&d_cell_S.CLog, (Ncell+1)*sizeof(int));
	cudaMalloc((void**)&d_cell_S.Point[0], (Ncell+1)*sizeof(int));
	cudaMalloc((void**)&d_cell_S.Point[1], (Ncell+1)*sizeof(int));
	cudaMalloc((void**)&d_cell_S.Point[2], (Ncell+1)*sizeof(int));
	cudaMalloc((void**)&d_cell_S.deltU[0][0], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.deltU[0][1], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.deltU[0][2], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.deltU[0][3], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.deltU[1][0], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.deltU[1][1], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.deltU[1][2], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.deltU[1][3], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.Umax[0], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.Umax[1], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.Umax[2], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.Umax[3], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.Umin[0], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.Umin[1], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.Umin[2], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.Umin[3], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.center[0], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.center[1], (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_cell_S.celledge[0], (Nnode+1)*sizeof(int));
	cudaMalloc((void**)&d_cell_S.celledge[1], (Nnode+1)*sizeof(int));
	cudaMalloc((void**)&d_cell_S.celledge[2], (Nnode+1)*sizeof(int));
	cudaMalloc((void**)&d_cell_S.neicell[0], (Nnode+1)*sizeof(int));
	cudaMalloc((void**)&d_cell_S.neicell[1], (Nnode+1)*sizeof(int));
	cudaMalloc((void**)&d_cell_S.neicell[2], (Nnode+1)*sizeof(int));
	
	cudaMalloc((void**)&d_Res_e_S.left[0], (Nedge+1)*sizeof(double));
	cudaMalloc((void**)&d_Res_e_S.left[1], (Nedge+1)*sizeof(double));
	cudaMalloc((void**)&d_Res_e_S.left[2], (Nedge+1)*sizeof(double));
	cudaMalloc((void**)&d_Res_e_S.left[3], (Nedge+1)*sizeof(double));
	cudaMalloc((void**)&d_Res_e_S.right[0], (Nedge+1)*sizeof(double));
	cudaMalloc((void**)&d_Res_e_S.right[1], (Nedge+1)*sizeof(double));
	cudaMalloc((void**)&d_Res_e_S.right[2], (Nedge+1)*sizeof(double));
	cudaMalloc((void**)&d_Res_e_S.right[3], (Nedge+1)*sizeof(double));

    cudaMalloc((void**)&d_deltT, (Ncell+1)*sizeof(double));
    cudaMalloc((void**)&d_drou , (Ncell/threadNum + 1)*sizeof(double));

	cudaMalloc((void**)&d_w_S.density   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w_S.density_U   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w_S.density_V   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w_S.density_E   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w_S.P   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w_S.T   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w_S.A   , (Ncell+1)*sizeof(double));  
    
	cudaMalloc((void**)&d_w0_S.density   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w0_S.density_U   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w0_S.density_V   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w0_S.density_E   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w0_S.P   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w0_S.T   , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_w0_S.A   , (Ncell+1)*sizeof(double));
	
	cudaMalloc((void**)&d_Wallw_S.density   , (WallBoundNum+1)*sizeof(double));
	cudaMalloc((void**)&d_Wallw_S.density_U   , (WallBoundNum+1)*sizeof(double));
	cudaMalloc((void**)&d_Wallw_S.density_V   , (WallBoundNum+1)*sizeof(double));
	cudaMalloc((void**)&d_Wallw_S.density_E   , (WallBoundNum+1)*sizeof(double));
	cudaMalloc((void**)&d_Wallw_S.P   , (WallBoundNum+1)*sizeof(double));
	cudaMalloc((void**)&d_Wallw_S.T   , (WallBoundNum+1)*sizeof(double));
	cudaMalloc((void**)&d_Wallw_S.A   , (WallBoundNum+1)*sizeof(double));
	
	cudaMalloc((void**)&d_Farw_S.density , (FarBoundNum+1)*sizeof(double));	
	cudaMalloc((void**)&d_Farw_S.density_U , (FarBoundNum+1)*sizeof(double));	
	cudaMalloc((void**)&d_Farw_S.density_V , (FarBoundNum+1)*sizeof(double));	
	cudaMalloc((void**)&d_Farw_S.density_E , (FarBoundNum+1)*sizeof(double));	
	cudaMalloc((void**)&d_Farw_S.P , (FarBoundNum+1)*sizeof(double));	
	cudaMalloc((void**)&d_Farw_S.T , (FarBoundNum+1)*sizeof(double));	
	cudaMalloc((void**)&d_Farw_S.A , (FarBoundNum+1)*sizeof(double));	
	
	cudaMalloc((void**)&d_Res_S.data[0]  , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_Res_S.data[1]  , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_Res_S.data[2]  , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_Res_S.data[3]  , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_R1_S.data[0]  , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_R1_S.data[1]  , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_R1_S.data[2]  , (Ncell+1)*sizeof(double));
	cudaMalloc((void**)&d_R1_S.data[3]  , (Ncell+1)*sizeof(double));
	
	cudaMalloc((void**)&d_edge_S.ELog  , 		(Nedge+1)*sizeof(int));
	cudaMalloc((void**)&d_edge_S.left_cell  , 	(Nedge+1)*sizeof(int));
	cudaMalloc((void**)&d_edge_S.right_cell  , 	(Nedge+1)*sizeof(int));
	cudaMalloc((void**)&d_edge_S.farfieldid  , 	(Nedge+1)*sizeof(int));
	cudaMalloc((void**)&d_edge_S.wallid  , 		(Nedge+1)*sizeof(int));
	cudaMalloc((void**)&d_edge_S.node1  , 		(Nedge+1)*sizeof(int));
	cudaMalloc((void**)&d_edge_S.node2  , 		(Nedge+1)*sizeof(int));
	cudaMalloc((void**)&d_edge_S.vectorn  , 	(Nedge+1)*sizeof(double));
	cudaMalloc((void**)&d_edge_S.vectorx  , 	(Nedge+1)*sizeof(double));
	cudaMalloc((void**)&d_edge_S.vectory  , 	(Nedge+1)*sizeof(double));	
	cudaMalloc((void**)&d_edge_S.midx  , 		(Nedge+1)*sizeof(double));
	cudaMalloc((void**)&d_edge_S.midy  , 		(Nedge+1)*sizeof(double));
}	

void struct2Array()
{
			
	for(int i=1; i< Nedge+1; i++)
	{
		edge_S.ELog[i] = edge[i].ELog;
		edge_S.left_cell[i] = edge[i].left_cell;
		edge_S.right_cell[i] = edge[i].right_cell;
		edge_S.farfieldid[i] = edge[i].farfieldid;
		edge_S.wallid[i] = edge[i].wallid;
		edge_S.node1[i] = edge[i].node1;
		edge_S.node2[i] = edge[i].node2;
		edge_S.vectorn[i] = edge[i].vectorn;
		edge_S.vectorx[i] = edge[i].vectorx;
		edge_S.vectory[i] = edge[i].vectory;
		edge_S.midx[i] = edge[i].midx;
		edge_S.midy[i] = edge[i].midy;
	}
	
	for(int i=1; i< Ncell+1; i++)
	{
		w0_S.density[i] = w[i].density;
		w0_S.density_U[i] = w[i].density_U;
		w0_S.density_V[i] = w[i].density_V;
		w0_S.density_E[i] = w[i].density_E;
		w0_S.P[i] = w[i].P;
		w0_S.T[i] = w[i].T;
		w0_S.A[i] = w[i].A;
		
		w_S.density[i] = w[i].density;
		w_S.density_U[i] = w[i].density_U;
		w_S.density_V[i] = w[i].density_V;
		w_S.density_E[i] = w[i].density_E;
		w_S.P[i] = w[i].P;
		w_S.T[i] = w[i].T;
		w_S.A[i] = w[i].A;
	}
	
	for(int i=1; i< Nnode+1; i++)
	{
		node_S.x[i] = node[i].x;
		node_S.y[i] = node[i].y;
	}
	
	for(int i=1; i< Ncell+1; i++)
	{
		cell_S.CLog[i] = cell[i].CLog;
		cell_S.Point[0][i] = cell[i].Point[0];
		cell_S.Point[1][i] = cell[i].Point[1];
		cell_S.Point[2][i] = cell[i].Point[2];
		cell_S.center[0][i] = cell[i].center[0];
		cell_S.center[1][i] = cell[i].center[1];	
		cell_S.celledge[0][i] = cell[i].neighbor[0].celledge;
		cell_S.celledge[1][i] = cell[i].neighbor[1].celledge;
		cell_S.celledge[2][i] = cell[i].neighbor[2].celledge;
		cell_S.neicell[0][i] = cell[i].neighbor[0].neicell;
		cell_S.neicell[1][i] = cell[i].neighbor[1].neicell;
		cell_S.neicell[2][i] = cell[i].neighbor[2].neicell;
	}
}

int ROE_SOLVER(W *w,EDGE *edge,double *CellArea,CELL *cell,NODE *node)
{
	int	   IR,i;//icell, iedge;
//	double FLUX[4],ds;



	struct timeval start,end,start_init;
	double	t1,t2,t3,t4,t5,t6,t7,t8,t_total;

	int		threadNum, blockNum;
	threadNum = MaxThreads;	
		
	step=0;
	Resd[step]=100.0;
	
	cpuMalloc(threadNum);
	gpuMalloc(threadNum);

	changeEdge();

	struct2Array();

	copyH2D();

	cudaPrintfInit();  

	gettimeofday(&start_init, NULL); 
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	while(step<MAXSTEP && Resd[step]>EPSL)
	{	
		step++;
		drou=0.0;
		
		cudaMemcpy(d_w0_S.density	, d_w_S.density	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_w0_S.density_U	, d_w_S.density_U	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_w0_S.density_V	, d_w_S.density_V	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_w0_S.density_E	, d_w_S.density_E	, (Ncell+1)*sizeof(double), cudaMemcpyDeviceToDevice);

		// RK START
		for(IR=0; IR<4; IR++)
		{

			gettimeofday(&start, NULL);      

            // Init
			blockNum  = (Ncell)/threadNum + 1;
 			
   			cudaThreadSynchronize();
   			gettimeofday(&end, NULL);  
   			t1+=((double)( 1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec ))/1000000.0;
   			


			gettimeofday(&start, NULL); 
			//compute the varialbels of the boundary cells
			blockNum = Nedge / threadNum + 1;
			kernel_Boundary<<< blockNum, threadNum>>>
					(d_w_S, d_Wallw_S, d_Farw_S, d_cell, d_edge_S,
					GAMA,UI,VI,AIN,TIN,VIN,R,PIS, Nedge);

			cudaThreadSynchronize();
			gettimeofday(&end, NULL);  
   			t2+=((double)( 1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec ))/1000000.0;


			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			gettimeofday(&start, NULL); 		
			//compute the gradient getting ready for interpolating
			blockNum  = (Ncell)/threadNum + 1;
			kernel_Gradient<<< blockNum, threadNum>>>
					(d_w_S,d_Wallw_S,d_Farw_S,d_cell,d_edge_S,d_CellArea,d_node_S, Ncell);	

 #if 0

			if(IR==0)
			{	
				copyD2H();
				
				for(int ii=0; ii<Ncell+1; ii++) 
					cout<<"deltT "<<ii<<" "<<deltT[ii]<<endl;
					
				for(int ii=0; ii<Ncell+1; ii++) 
					cout<<"Res "<<ii<<" "<<Res_S.data[0][ii]
									<<" "<<Res_S.data[1][ii]
									<<" "<<Res_S.data[2][ii]
									<<" "<<Res_S.data[3][ii]<<endl;
									
				for(int ii=0; ii<Ncell+1; ii++) 
					cout<<"w "<<ii<<" "<<w_S.density[ii]
							<<" "<<w_S.density_U[ii]
							<<" "<<w_S.density_V[ii]
							<<" "<<w_S.density_E[ii]
							<<" "<<w_S.P[ii]
							<<" "<<w_S.T[ii]
							<<" "<<w_S.A[ii]<<endl;

				for(int ii=0; ii<Ncell+1; ii++) 
					cout<<"w0 "<<ii<<" "<<w0_S.density[ii]
							<<" "<<w0_S.density_U[ii]
							<<" "<<w0_S.density_V[ii]
							<<" "<<w0_S.density_E[ii]
							<<" "<<w0_S.P[ii]
							<<" "<<w0_S.T[ii]
							<<" "<<w0_S.A[ii]<<endl;
											
				for(int ii=0; ii<WallBoundNum+1; ii++) 
					cout<<"Wallw "<<ii<<" "<<Wallw_S.density[ii]
							<<" "<<Wallw_S.density_U[ii]
							<<" "<<Wallw_S.density_V[ii]
							<<" "<<Wallw_S.density_E[ii]
							<<" "<<Wallw_S.P[ii]
							<<" "<<Wallw_S.T[ii]
							<<" "<<Wallw_S.A[ii]<<endl;
				
				for(int ii=0; ii<FarBoundNum+1; ii++) 
					cout<<"Farw "<<ii<<" "<<Farw_S.density[ii]
							<<" "<<Farw_S.density_U[ii]
							<<" "<<Farw_S.density_V[ii]
							<<" "<<Farw_S.density_E[ii]
							<<" "<<Farw_S.P[ii]
							<<" "<<Farw_S.T[ii]
							<<" "<<Farw_S.A[ii]<<endl;
				
				for(int ii=0; ii<Ncell+1; ii++) 
					cout<<"cell "<<ii<<" "<<cell[ii].CLog
							<<" "<<cell[ii].Point[0]<<cell[ii].Point[1]<<cell[ii].Point[2]
							<<" "<<cell[ii].center[0]<<cell[ii].center[1]
							<<" "<<cell[ii].neighbor[0].celledge<<cell[ii].neighbor[0].neicell
							<<" "<<cell[ii].neighbor[1].celledge<<cell[ii].neighbor[1].neicell
							<<" "<<cell[ii].neighbor[2].celledge<<cell[ii].neighbor[2].neicell
							<<" "<<cell[ii].deltU[0][0]<<cell[ii].deltU[0][1]<<cell[ii].deltU[0][2]<<cell[ii].deltU[0][3]
							<<" "<<cell[ii].deltU[1][0]<<cell[ii].deltU[1][1]<<cell[ii].deltU[1][2]<<cell[ii].deltU[1][3]
							<<" "<<cell[ii].Umax[0]<<cell[ii].Umax[1]<<cell[ii].Umax[2]<<cell[ii].Umax[3]
							<<" "<<cell[ii].Umin[0]<<cell[ii].Umin[1]<<cell[ii].Umin[2]<<cell[ii].Umin[3]<<endl;				
				
				for(int ii=0; ii<Nedge+1; ii++)
					cout<<"edge "<<ii<<" "<<edge_S.ELog[ii]
							<<" "<<edge_S.left_cell[ii]
							<<" "<<edge_S.right_cell[ii]
							<<" "<<edge_S.vectorx[ii]
							<<" "<<edge_S.vectory[ii]
							<<" "<<edge_S.vectorn[ii]
							<<" "<<edge_S.node1[ii]
							<<" "<<edge_S.node2[ii]
							<<" "<<edge_S.midx[ii]
							<<" "<<edge_S.midy[ii]
							<<" "<<edge_S.farfieldid[ii]
							<<" "<<edge_S.wallid[ii]<<endl;
							
				for(int ii=0; ii<Ncell+1; ii++)
					cout<<"CellArea "<<ii<<" "<<CellArea[ii]<<endl;
					
				for(int ii=0; ii<Nnode+1; ii++)
					cout<<"node "<<ii<<" "<<node[ii].NLog	
							<<" "<<node[ii].x
							<<" "<<node[ii].y
							<<" "<<node[ii].ROU
							<<" "<<node[ii].U
							<<" "<<node[ii].V
							<<" "<<node[ii].E
							<<" "<<node[ii].P
							<<" "<<node[ii].T
							<<" "<<node[ii].A
							<<" "<<node[ii].Mach
							<<" "<<node[ii].RoundArea<<endl;
							
				return 0;
			}
#endif		
					
			cudaThreadSynchronize();
			gettimeofday(&end, NULL);  
   			t3+=((double)( 1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec ))/1000000.0;
   			

			gettimeofday(&start, NULL); 	
			// compute flux
			blockNum = Nedge / threadNum + 1;
			kernel_Flux2<<< blockNum, threadNum>>>
					(d_w_S,d_Wallw_S,d_Farw_S,d_CellArea,d_edge_S,d_cell,d_Res_e_S,GAMA,Nedge);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			kernel_deltT<<< blockNum, threadNum>>>
					(d_w_S,d_Wallw_S,d_Farw_S,d_deltT_e_S,d_edge_S,GAMA,Nedge);

					
			cudaThreadSynchronize();
			gettimeofday(&end, NULL);  
   			t4+=((double)( 1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec ))/1000000.0;
  
			
			gettimeofday(&start, NULL);
			blockNum = Ncell / threadNum + 1;
			kernel_Flux_e2c<<< blockNum, threadNum>>>
					(d_deltT_e_S, d_deltT, 
					d_Res_e_S, d_Res_S,
					 d_cell, d_edge_S,d_CellArea,CFL, Ncell);	
					 	
			cudaThreadSynchronize();
			gettimeofday(&end, NULL);  
   			t5+=((double)( 1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec ))/1000000.0;


			gettimeofday(&start, NULL);
			// timestep
//			blockNum  = (Ncell)/threadNum + 1;
//			kernel_TimeStep<<< blockNum, threadNum>>>
//						(d_deltT,d_CellArea,CFL,Ncell);
			// dt
//			cudaThreadSynchronize();

//			cudaMemcpy(deltT, d_deltT, blockNum*sizeof(double), cudaMemcpyDeviceToHost);
//			dt = 10000000.0;
//			for(i=0; i<blockNum; i++)
//			{
//				if(dt>deltT[i]) dt=deltT[i];
//			}

			Smooth(blockNum, threadNum );
//			cudaThreadSynchronize();
			gettimeofday(&end, NULL);  
   			t6+=((double)( 1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec ))/1000000.0;

			gettimeofday(&start, NULL);
			// RK step
			blockNum  = (Ncell)/threadNum + 1;
			kernel_RKStep<<< blockNum, threadNum>>>
					(d_w_S, d_w0_S, RK[IR],d_deltT, d_Res_S, 
					d_CellArea, GAMA, R, Ncell);

			cudaThreadSynchronize();

			// drou		
			if(IR==3)
			{
				kernel_drou<<< blockNum, threadNum>>>
						(d_w_S, d_w0_S, d_drou, Ncell);
//				cudaThreadSynchronize();
				cudaMemcpy(rou, d_drou, blockNum*sizeof(double), cudaMemcpyDeviceToHost);
				drou=0.0;
				for(i=0; i<blockNum; i++)
				{
					drou+=rou[i];
				}
//cudaPrintfDisplay(stdout, false);
//cudaPrintfEnd(); 
//return 0;
			

							
			}
			gettimeofday(&end, NULL);  
   			t8+=((double)( 1000000*(end.tv_sec-start.tv_sec)+end.tv_usec-start.tv_usec ))/1000000.0;
			cudaThreadSynchronize();

		}
		
		if(step == 1)
			EPSM = drou;
		
		Resd[step]=sqrt((drou+1.0E-32)/EPSM);
		
		printf("%d,%1.20lf\n",step,Resd[step]);
	}
	gettimeofday(&end, NULL); 
	t_total=((double)( 1000000*(end.tv_sec-start_init.tv_sec)+end.tv_usec-start_init.tv_usec ))/1000000.0;
	
	cout<<t1<<endl
		<<t2<<endl
		<<t3<<endl
		<<t4<<endl
		<<t5<<endl
		<<t6<<endl
		<<t7<<endl
		<<t8<<endl
		<<"total time: "<<t_total<<endl;
	
	copyD2H();
	
	free(w0);
//	free(Res);
	free(deltT);
	free(Wallw);
	free(Farw);
	free(rou);
	
	cudaFree(d_drou);
	cudaFree(d_deltT);
	cudaFree(d_w0);
//	cudaFree(d_Res);
	cudaFree(d_Wallw);
	cudaFree(d_Farw);
	
	cudaMemcpy(cell, d_cell, (Ncell+1)*sizeof(CELL), cudaMemcpyDeviceToHost);
	cudaMemcpy(node, d_node, (Ncell+1)*sizeof(CELL), cudaMemcpyDeviceToHost);
	cudaMemcpy(edge, d_edge, (Ncell+1)*sizeof(CELL), cudaMemcpyDeviceToHost);
	cudaMemcpy(w   , d_w   , (Ncell+1)*sizeof(CELL), cudaMemcpyDeviceToHost);
	cudaMemcpy(CellArea, d_CellArea, (Ncell+1)*sizeof(CELL), cudaMemcpyDeviceToHost);
    
 	cudaPrintfDisplay(stdout, false);
    cudaPrintfEnd(); 
	return 0;
}
