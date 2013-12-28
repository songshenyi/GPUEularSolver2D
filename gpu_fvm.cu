#include "gpu_fvm.cuh"

using namespace std;

int    Ncell,Nnode,Nedge,MAXSTEP,WallNum,step,WallBoundNum,FarBoundNum;
double MA,ALPHA,GAMA,PIN,TIN,CFL,EPSL,UI,VI;
double ROUIN,AIN,VIN,PIS;
double R=287.06;
double PI=3.14159265358979;
double RK[4]={0.0833,0.2069,0.4265,1.0};
double *CellArea;
double *Resd;


CELL   *d_cell;
NODE   *d_node;
EDGE   *d_edge;
W      *d_w;
double	*d_CellArea;


CELL   *cell;
NODE   *node;
EDGE   *edge,*WallEdge;
W      *w;
	
int main()
{	
//	clock_t start, finish;

	inputMesh(&cell,&node);

	w			=(W    *)calloc(Ncell+1,sizeof(W));
	edge		=(EDGE *)calloc(Ncell*3,sizeof(EDGE));
	Resd		= (double *)calloc((MAXSTEP+1),sizeof(double));
	CellArea	= (double *)calloc((Ncell+1),sizeof(double)); 
	WallEdge	=(EDGE *)calloc(Ncell+1   ,sizeof(EDGE));

    Initialization(w);
	CalculationMeshGeo(cell,node,edge,WallEdge);
	
	Cuda_Init();
	

	ROE_SOLVER(w,edge,CellArea,cell,node);

	
	Node_Computing(w,CellArea,cell,node,edge);
	Output(node,cell,step,WallEdge);
	
	free(cell);
	free(node);
	free(edge);
	free(w);
	free(WallEdge);
	free(CellArea);
	free(Resd);
	
	cudaFree(d_cell);
	cudaFree(d_node);
	cudaFree(d_edge);
	cudaFree(d_w);
	cudaFree(d_CellArea);


	printf("All Computation's Finished!!\n");
	return 0;
}


int Node_Computing(W *w,double *CellArea,CELL *cell,NODE *node,EDGE *edge)
{

	int i,icell,inode;
	double pcell,tcell,acell;

	WallNum=0;

	for(inode=1;inode<=Nnode;inode++)
	{
		node[inode].RoundArea=0.0;
		node[inode].P=0;
		node[inode].ROU=0;
		node[inode].T=0;
		node[inode].U=0;
		node[inode].V=0;
		node[inode].E=0;

		if(node[inode].NLog==2)
			WallNum++;
	}

	for(icell=1;icell<=Ncell;icell++)
	{
		tcell=w[icell].T;
		pcell=w[icell].P;
		acell=w[icell].A;

		for(i=0;i<3;i++)
		{
			 node[cell[icell].Point[i]].ROU		 +=w[icell].density  *CellArea[icell];
			 node[cell[icell].Point[i]].U		 +=w[icell].density_U*CellArea[icell];
			 node[cell[icell].Point[i]].V        +=w[icell].density_V*CellArea[icell];
			 node[cell[icell].Point[i]].E        +=w[icell].density_E*CellArea[icell];

			 node[cell[icell].Point[i]].T		 +=tcell*CellArea[icell];
			 node[cell[icell].Point[i]].P        +=pcell*CellArea[icell];
			 node[cell[icell].Point[i]].A        +=acell*CellArea[icell];

			 node[cell[icell].Point[i]].RoundArea+=CellArea[icell];
		}
	}

	for(inode=1;inode<=Nnode;inode++)
	{
		node[inode].ROU/=node[inode].RoundArea;
		node[inode].U  /=node[inode].RoundArea;
		node[inode].V  /=node[inode].RoundArea;
		node[inode].E  /=node[inode].RoundArea;

		node[inode].U  /=node[inode].ROU;
		node[inode].V  /=node[inode].ROU;
		
		node[inode].P /=node[inode].RoundArea;
		node[inode].T /=node[inode].RoundArea;
		node[inode].A /=node[inode].RoundArea;

		node[inode].Mach=sqrt( node[inode].U*node[inode].U+node[inode].V*node[inode].V )/node[inode].A;
	}
	return 0;
}

int Output(NODE *node,CELL *cell,int step,EDGE *WallEdge)
{
	FILE *fp;
	double PCoe;
	int inode,icell,i;

    if((fp=fopen("unresult.plt","w"))==NULL)
	{
		printf("\nCannot open file result.dat\n");
		exit(0);
	}

	fprintf(fp,"TITLE= \"Data\"\n");
	fprintf(fp,"VARIABLES = \"X\" \"Y\" \"Rou\" \"U\" \"V\" \"P\" \"T\" \"MA\" \n");
	fprintf(fp,"ZONE T=\"AirfoilMesh\"\n");
	fprintf(fp,"N=%d,E=%d, ZONETYPE=FETriangle\n",Nnode,Ncell);
	fprintf(fp,"DATAPACKING=POINT\n DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE)\n");

	for(inode=1;inode<=Nnode;inode++)
		fprintf(fp,"%20.10f %20.10f %20.10f %20.10f %20.10f %20.10f %20.10f %20.10f \n",node[inode].x,node[inode].y,node[inode].ROU,
														node[inode].U,node[inode].V,node[inode].P,node[inode].T,node[inode].Mach);

	for(icell=1;icell<=Ncell;icell++)
	{	
		for(i=0;i<3;i++)
			fprintf(fp,"%10d",cell[icell].Point[i]);

		fprintf(fp,"\n");		
	}

	fclose(fp);

	fp=fopen("unresidual.plt","w");
	fprintf(fp,"TITLE     = \"Dataset\"\nVARIABLES = \"Time\" \"Resid\" \n");
	fprintf(fp,"ZONE I=%d J=%d f=point \n",step,1);

	for(i=1;i<=step;i++)
	{
		if(i%100==0)
		fprintf(fp,"%d %20.10lf\n",i,Resd[i]); 
	}

	fclose(fp);

	fp=fopen("unairfoilpress.plt","w");
	fprintf(fp,"TITLE     = \"Dataset\"\nVARIABLES = \"Position\" \"Pressure\" \n");
	fprintf(fp,"ZONE N=%d E=%d F=FEPOINT ET=Triangle\n",WallBoundNum*2,WallBoundNum);                    
      
	for(i=1;i<=WallBoundNum;i++){
		PCoe=-2.0*(node[WallEdge[i].node1].P-PIN/(ROUIN*VIN*VIN));
		fprintf(fp,"%20.10lf %20.10lf\n",node[WallEdge[i].node1].x,PCoe);
		PCoe=-2.0*(node[WallEdge[i].node2].P-PIN/(ROUIN*VIN*VIN));
		fprintf(fp,"%20.10lf %20.10lf\n",node[WallEdge[i].node2].x,PCoe);
		WallEdge[i].node1=2*i-1;
		WallEdge[i].node2=2*i;
	}

	for(i=1;i<=WallBoundNum;i++)
		fprintf(fp,"%d %d %d\n",WallEdge[i].node1,WallEdge[i].node2,WallEdge[i].node1);

	fclose(fp);

	return 0;
}
