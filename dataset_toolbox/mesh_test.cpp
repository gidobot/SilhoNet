#include <math.h>
#include <matrix.h>
#include <mex.h>

#define min(a, b) (((a) <= (b)) ? (a) : (b))
#define max(a, b) (((a) >= (b)) ? (a) : (b))

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
      int imgw, imgh;
      
      const mwSize *dimsv;
      double *vertex_in;
      mxLogical *outArray;
      imgh = mxGetScalar(prhs[1]);
      imgw = mxGetScalar(prhs[2]);

      dimsv = mxGetDimensions(prhs[0]);

      int dimsv_1 = (int) dimsv[0];

      vertex_in = mxGetPr(prhs[0]);
      
      plhs[0] = mxCreateLogicalMatrix(imgh, imgw);
      outArray = mxGetLogicals(plhs[0]);
      
      double x1,y1,x2,y2,x3,y3, v0x, v0y, v1x, v1y, v2x, v2y;
      double minx, miny, maxx, maxy;
      double dot00, dot01, dot02, dot11, dot12, invdenom;
      double u,v;
      for (int i = 0; i < dimsv_1; i++)
      {
         x1 = round(vertex_in[1 * dimsv_1 + i]);  
         y1 = round(vertex_in[0 * dimsv_1 + i]);
         
         x2 = round(vertex_in[3 * dimsv_1 + i]);
         y2 = round(vertex_in[2 * dimsv_1 + i]);
         
         x3 = round(vertex_in[5 * dimsv_1 + i]);
         y3 = round(vertex_in[4 * dimsv_1 + i]);

         //if (x1 <= 0 || y1 <= 0 || x2 <= 0 || y2 <= 0 || x3 <= 0 || y3 <= 0 || y1 >= imgh || x1 >= imgw || y2 >= imgh || x2 >= imgw || y3>=imgh || x3 >=imgw)
         //{
         //  continue;
         //}
                   
         minx = min(min(x1,x2),x3);
         if (minx == 0)
         {
           mexPrintf("****** %d %d %d ******\n", x1, x2, x3);
         }
         miny = min(min(y1,y2),y3);

         maxx = max(max(x1,x2),x3);
         maxy = max(max(y1,y2),y3);

         for (int mm = miny; mm <= maxy; mm=mm+1)
            for (int nn = minx; nn <= maxx; nn=nn+1)
            {
               v0x = x3 - x1;
               v0y = y3 - y1;

               v1x = x2 - x1;
               v1y = y2 - y1;
               
               v2x = double(nn) - x1;
               v2y = double(mm) - y1;
               
               dot00 = v0x * v0x + v0y * v0y;
               dot01 = v0x * v1x + v0y * v1y;
               dot02 = v0x * v2x + v0y * v2y;
               dot11 = v1x * v1x + v1y * v1y;
               dot12 = v1x * v2x + v1y * v2y;

               invdenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
               u = (dot11 * dot02 - dot01 * dot12) * invdenom;
               v = (dot00 * dot12 - dot01 * dot02) * invdenom;
               if ((u >= 0) && (v >= 0) && (u + v <= 1))
               {
                  //if (((nn-1)*imgh)+mm <= 0 || ((nn-1)*imgh)+mm > (imgh) * (imgw)-1)
	                if (nn < 1 || nn >= imgw || mm < 0 || mm > imgh)
                  {  
                     //mexPrintf("%d %d %d %d ****** \n", ((nn-1)*imgh)+mm, nn, imgh, mm);
                     continue;
                  }
                  outArray[((nn-1)*imgh)+mm] = 1;
               }
            }


      }
                                
}
