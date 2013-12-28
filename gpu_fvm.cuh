
#include "stdio.h"
#include "stdlib.h"
#include "cmath"
#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>

using namespace std;


typedef struct Neighbor
{
	int celledge;
	int neicell;
}Neighbor;

typedef struct CELL 
{
	int CLog;
	int Point[3];
	double deltU[2][4];
	double Umax[4],Umin[4];
	double center[2];
	Neighbor neighbor[3];
}CELL;

typedef struct CELL_S 
{
	int *CLog;
	int *Point[3];
	double *deltU[2][4];
	double *Umax[4],*Umin[4];
	double *center[2];
	int *celledge[3];
	int	*neicell[3];
}CELL_S;

typedef struct NODE
{
	int NLog,Nlog1;
	double x;
	double y;
	double ROU,U,V,E;
	double P,T,A;
	double Mach;
	double RoundArea;
}NODE;

typedef struct NODE_S
{
	int *NLog,*Nlog1;
	double *x;
	double *y;
	double *ROU,*U,*V,*E;
	double *P,*T,*A;
	double *Mach;
	double *RoundArea;
}NODE_S;

typedef struct EDGE
{
	int ELog,Elog1;
	int left_cell;
	int right_cell;
	int farfieldid;
	int wallid;
	int node1;
	int node2;
	double vectorn;
	double vectorx;
	double vectory;
	double midx,midy;

}EDGE;

typedef struct EDGE_S
{
	int *ELog;
	int *left_cell;
	int *right_cell;
	int *farfieldid;
	int *wallid;
	int *node1;
	int *node2;
	double *vectorn;
	double *vectorx;
	double *vectory;
	double *midx,*midy;

}EDGE_S;


typedef struct W
{
	double density;
	double density_U;
	double density_V;
	double density_E;
	double P;
	double T;
	double A;
	double B;
}W;

typedef struct W_S
{
	double *density;
	double *density_U;
	double *density_V;
	double *density_E;
	double *P;
	double *T;
	double *A;
	double *B;
}W_S;

typedef struct RESD
{
	double data[4];
}RESD;

typedef struct RESD_S
{
	double *data[4];
}RESD_S;

typedef struct DT_E
{
	double left;
	double right;
}DT_E;

typedef struct DT_E_S
{
	double *left;
	double *right;
}DT_E_S;

typedef struct DR_E
{
	double left[4];
	double right[4];
}DR_E;

typedef struct DR_E_S
{
	double *left[4];
	double *right[4];
}DR_E_S;

extern int    Ncell,Nnode,Nedge,MAXSTEP,WallNum,step,WallBoundNum,FarBoundNum;
extern double MA,ALPHA,GAMA,PIN,TIN,CFL,EPSL,UI,VI;
extern double ROUIN,AIN,VIN,PIS;
extern double R;
extern double PI;
extern double RK[4];
extern double *CellArea;
extern double *Resd;

extern CELL   *cell;
extern NODE   *node;
extern EDGE   *edge,*WallEdge;
extern W      *w;


extern CELL   *d_cell;
extern NODE   *d_node;
extern EDGE   *d_edge;
extern W      *d_w;
extern double	*d_CellArea;

extern W *w0;
extern double EPSM;
extern double *deltT;
extern double drou,dt,*rou;
extern RESD 	*Res;	
extern W      *Wallw,*Farw;

extern double	*d_deltT;
extern double	*d_drou;
extern W		*d_w0;
extern RESD	*d_Res, *d_R1;
extern W		*d_Wallw, *d_Farw;


int inputMesh(CELL **cell,NODE **node);

int CalculationPTA(W &WW);

int Initialization(W *w);

double Calc_area(double x1,double y1,double x2,double y2,double x3,double y3);

int CalculationMeshGeo(CELL *cell,NODE *node,EDGE *edge,EDGE *WallEdge);

double Calc_SRI(W &LW,W &RW,EDGE &Iedge);

int Calc_Function(W &w,double &A,double &U,double &V,double &H,double &P);

int Far_Boundary(W *w,W &w0,W &DW,EDGE &Iedge);

int  Boundary(W *w,W *Wallw,W *Farw,CELL *cell,EDGE *edge);

int Gradient(W *w,W *Wallw,W *Farw,CELL *cell,EDGE *edge,double *CellArea,NODE *node);

int Limiter(W *w,EDGE &Iedge,int Cellnum,CELL *cell,EDGE *edge,double fi[4],double delt2[4]);

int LBJinterpolate(W *w,EDGE &Iedge,CELL *cell,EDGE *edge,W &LW,W &RW,double *CellArea,NODE *node);

int ROE_SCHEME(W &LW,W &RW,double nx,double ny,double FLUX[4]);

int Flux(W *w,W *Wallw,W *Farw,EDGE &Iedge,double FLUX[4],double *CellArea,double *deltT,int iedge,CELL *cell,EDGE *edge,NODE *node);

// int Smooth(W *w,RESD *Res,EDGE *edge,CELL *cell);

double TimeStep(double *deltT,double *CellArea);

int ROE_SOLVER(W *w,EDGE *edge,double *CellArea,CELL *cell,NODE *node);

int Node_Computing(W *w,double *CellArea,CELL *cell,NODE *node,EDGE *edge);

int Output(NODE *node,CELL *cell,int step,EDGE *WallEdge);

int	Cuda_Init();
