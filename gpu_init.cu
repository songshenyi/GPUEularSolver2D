#include "gpu_fvm.cuh"
#include <cuda.h>
//#include "cuPrintf.cu"

int inputMesh(CELL **cell,NODE **node)
{
	FILE *fp;
	int icell,inode;

	if( (fp=fopen("2822-3k.dat","r"))==NULL )
	{
		printf("\nCannot open the mesh data file!\n");
		exit(0);
	}

	fscanf(fp,"%d %d",&Nnode,&Ncell);

	*cell		=(CELL *)calloc(Ncell+1,sizeof(CELL));
	*node		=(NODE *)calloc(Nnode+1,sizeof(NODE));

	for(inode=1;inode<=Nnode;inode++)
		fscanf(fp,"%lf",&(*node)[inode].x);

	for(inode=1;inode<=Nnode;inode++)
		fscanf(fp,"%lf",&(*node)[inode].y);

	for(icell=1;icell<=Ncell;icell++)
		for(int i=0;i<3;i++)
		{
			fscanf(fp,"%d",&(*cell)[icell].Point[i]);
//			&(*cell+icell)->Point[i]--;
		}
	fclose(fp);

	if((fp=fopen("input.dat","r"))==NULL)
	{
		printf("\nCannot open input file!\n");
		exit(0);
	}

	fscanf(fp,"%lf %lf %lf %lf %lf %lf %lf %d",&MA,&ALPHA,&GAMA,&PIN,&TIN,&CFL,&EPSL,&MAXSTEP);
	fclose(fp);

	printf("inputData finished!\n");

	return 0;
}

int CalculationPTA(W &WW){
	double U,V,E;
	
	U=WW.density_U/WW.density;
	V=WW.density_V/WW.density;
	E=WW.density_E/WW.density-0.5*(U*U+V*V);
	
	WW.P=(GAMA-1.0)*WW.density*E;
	WW.A=sqrt(GAMA*WW.P/WW.density);
	WW.T=(GAMA-1.0)*E/R;

	return 0;
}

int Initialization(W *w)
{	
	int icell;
	double rou_U,rou_V,rou_E;
	
	ALPHA=ALPHA*PI/180.0;
	AIN=sqrt(GAMA*R*TIN);
	VIN=MA*AIN;
	ROUIN=PIN/(R*TIN);

	UI=cos(ALPHA);
	VI=sin(ALPHA);

	rou_U=cos(ALPHA);
	rou_V=sin(ALPHA);
	rou_E=R*TIN/(VIN*VIN)/(GAMA-1.0)+0.50;

	for(icell=1;icell<=Ncell;icell++)
	{
		w[icell].density  =1.0;
		w[icell].density_U=rou_U;
		w[icell].density_V=rou_V;
		w[icell].density_E=rou_E;

		CalculationPTA(w[icell]);	
	}

	PIS=w[1].P;

	return 0;
}

double Calc_area(double x1,double y1,double x2,double y2,double x3,double y3)
{
	return(0.5*((y3-y1)*(x2-x1)-(y2-y1)*(x3-x1)));
}

int CalculationMeshGeo(CELL *cell,NODE *node,EDGE *edge,EDGE *WallEdge)
{
	int icell,iedge;
	int i,j,nn;
	int Findedge,Findcell,ci;
	int IP[3]={1,2,0};

	for(icell=1;icell<=Ncell;icell++){
		CellArea[icell]=Calc_area(node[cell[icell].Point[0]].x,node[cell[icell].Point[0]].y,
								  node[cell[icell].Point[1]].x,node[cell[icell].Point[1]].y,
								  node[cell[icell].Point[2]].x,node[cell[icell].Point[2]].y);

		if( CellArea[icell]<=0.0)printf("CellArea<0! %d %f\n",icell, CellArea[icell]);

		cell[icell].center[0]=(node[cell[icell].Point[0]].x+node[cell[icell].Point[1]].x+node[cell[icell].Point[2]].x)/3.0;
		cell[icell].center[1]=(node[cell[icell].Point[0]].y+node[cell[icell].Point[1]].y+node[cell[icell].Point[2]].y)/3.0;
	}

	FarBoundNum=0;
	WallBoundNum=0;
    Nedge=0;
	for(icell=1;icell<=Ncell;icell++)
	{

		if(icell%100 ==0)printf("icell %d\n",icell);
		for(i=0;i<3;i++)
		{			
		    int ie1=IP[i];
			int ie2=IP[ie1];
			int N1=cell[icell].Point[ie1];
			int N2=cell[icell].Point[ie2];

			Findedge=0;

			for(iedge=1;iedge<=Nedge;iedge++)
			{
				if( ( edge[iedge].node1==N1 && edge[iedge].node2==N2 ) || ( edge[iedge].node1==N2 && edge[iedge].node2==N1 ) )
				{
					Findedge=1;
				    break;
				}
			}

			cell[icell].neighbor[i].celledge=iedge;

			if(Findedge==0)
			{
			  Nedge++;		  
			  edge[Nedge].left_cell=icell;
			  edge[Nedge].node1=N1;       
			  edge[Nedge].node2=N2;

			  edge[Nedge].vectorx=node[N1].x-node[N2].x;
			  edge[Nedge].vectory=node[N2].y-node[N1].y;

			  edge[Nedge].midx   = 0.5*(node[N1].x+node[N2].x);
			  edge[Nedge].midy   = 0.5*(node[N1].y+node[N2].y);

			  edge[Nedge].vectorn= sqrt(edge[Nedge].vectorx*edge[Nedge].vectorx + edge[Nedge].vectory*edge[Nedge].vectory);

			  Findcell=0;

				  for(ci=icell+1;ci<=Ncell;ci++)
				  {
					for(j=0;j<3;j++)
					{
					  ie1=IP[j];
					  ie2=IP[ie1];
					  int NN1=cell[ci].Point[ie1];
					  int NN2=cell[ci].Point[ie2];

					  if( (NN1==N1&&NN2==N2) || (NN1==N2&&NN2==N1) )
					  { 
						  Findcell=1;  
						  break;
					  }
					}

					if(Findcell==1)
						break;
				  }

				  if(Findcell==1){
					 edge[iedge].right_cell=ci;
					 edge[iedge].ELog=0;
					 }
			
				  else{
					  edge[iedge].right_cell=-1;	
					  
					  if( fabs(edge[iedge].midx) <2.0 && fabs(edge[iedge].midy) <2.0){
						  edge[iedge].ELog=2;
						  node[edge[iedge].node1].NLog=2;
						  node[edge[iedge].node2].NLog=2;
						  WallBoundNum++;
						  edge[iedge].wallid=WallBoundNum;
						  WallEdge[WallBoundNum]=edge[iedge];
					  }

					  else{
						  edge[iedge].ELog=1;
						  FarBoundNum++;
						  edge[iedge].farfieldid=FarBoundNum;
						}
					  }
			}

		}

		nn=3;

		for(i=0;i<3;i++){
			cell[icell].neighbor[i].neicell = edge[cell[icell].neighbor[i].celledge].left_cell == icell ?
												edge[cell[icell].neighbor[i].celledge].right_cell :
												edge[cell[icell].neighbor[i].celledge].left_cell;
			
			if(cell[icell].neighbor[i].neicell == -1){
				cell[icell].neighbor[i].neicell *= edge[cell[icell].neighbor[i].celledge].ELog;
				nn--;
			}
		}

		cell[icell].CLog=nn;
	}	

	printf("Mesh data Computing's finished!!\n");

	return 0;
}

bool InitGPUSet()  
{  
    char GPU[100] = "GPU: ";  
    cudaDeviceProp tCard;  
    int num = 0;  
    if(cudaSuccess == cudaGetDeviceCount(&num))  
    {  
        for(int i = 0; i < num; ++ i)  
        {  
            cudaSetDevice(i);  
            cudaGetDeviceProperties(&tCard, i);  
            puts(strcat(GPU , tCard.name));//返回的就是链接后的结果,也为其的嵌套使用提供了条件  
        }  
    }
    else  return false;  
    return true;  
}  


int	Cuda_Init()
{
//	if(!InitGPUSet())  puts("device is not ready!");  
	
	cudaSetDevice(2); 
/*		cudaPrintfInit(); 
	displayGPU_demo<<<2, 3>>>();  	
	        cudaPrintfDisplay(stdout, true);//true输出是哪一个block的第几个thread在执行本条输出语句，形如：[blockID, threadID]；false不输出  
        cudaPrintfEnd();  
*/
        	
	cudaMalloc(&d_cell, (Ncell+1)*sizeof(CELL));
	cudaMalloc(&d_node, (Ncell+1)*sizeof(NODE));
	cudaMalloc(&d_edge, (Nedge+1)*sizeof(EDGE));
	cudaMalloc(&d_w   , (Ncell+1)*sizeof(W   ));
	cudaMalloc(&d_CellArea, (Ncell+1)*sizeof(double));
	
	cudaMemcpy(d_cell, cell, (Ncell+1)*sizeof(CELL), cudaMemcpyHostToDevice);
	cudaMemcpy(d_node, node, (Nnode+1)*sizeof(NODE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edge, edge, (Nedge+1)*sizeof(EDGE), cudaMemcpyHostToDevice);
	cudaMemcpy(d_w   , w   , (Ncell+1)*sizeof(W), cudaMemcpyHostToDevice);
	cudaMemcpy(d_CellArea, CellArea, (Ncell+1)*sizeof(double), cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
	
	return 0;
}


