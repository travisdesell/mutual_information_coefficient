//STL Container Includes
#include <vector>
#include <tr1/unordered_map>
#include <deque>
#include <string>

//Utilities
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <sys/time.h>
#include <sys/wait.h>

//Number Limits
#include <limits>
using namespace std;

//Thread Includes
#include <sys/ipc.h>
#include <semaphore.h>
#include <pthread.h>

//Thread Specifics
sem_t *semDuo;

#define INT static_cast<int>
#define FLOAT static_cast<float>
#define DOUBLE static_cast<double>

#define HASH_SALT numeric_limits<int>::max()/6

#define NORMALTEXT  "\033[22;32m"
#define MAGTEXT     "\033[22;35m"
struct Point{
    int X, Y;
};
struct Grid{
    int X,Y;
};

//Other Containers
typedef tr1::unordered_map<int,int> HashMap;
//1D Vectors
typedef vector<float>  floatVector;
typedef vector<int>    intVector;
typedef vector<string> stringVector;
typedef vector<Point>  pointVector;

//2D Vectors
typedef vector<intVector >   int2DVector;
typedef vector<floatVector > float2DVector;

//tr1 Hash Functions
typedef tr1::hash<int> intHashFcn;


Point swapXY(Point source){
    Point dest;
    dest.X = source.Y, dest.Y = source.X;
    return dest;
}

struct pointXCompare{
    bool operator ()(Point const &left, Point const &right) const{
        if(left.X < right.X) return true;
        if(left.X > right.X) return false;
        
        if(left.Y < right.Y) return true;
        if(left.Y > right.Y) return false;
        
        return false;
    }
};

struct pointYCompare{
    bool operator ()(Point const &left, Point const &right) const{
        if(left.Y < right.Y) return true;
        if(left.Y > right.Y) return false;
        
        if(left.X < right.X) return true;
        if(left.X > right.X) return false;
        
        return false;
    }
};

class pointHashFcn{
public:
    pointHashFcn(int salt) : salt(salt) {}
    float operator()(Point &in){//Salted Cantor's Enumeration of Pairs
        return myHashFcn(salt + INT(pow((in.X + in.Y),2) + (in.X - in.Y)) >> 1);
    }
private:
    int salt;
    intHashFcn myHashFcn;
};

class MICHeuristic{
public:
    static floatVector approxMaxMI(pointVector &xVec, pointVector &yVec,
    int maxXPartitions, int yPartitions, double numClumpsFactor){
        if((maxXPartitions < 2 || (yPartitions < 2) || (double)maxXPartitions * numClumpsFactor < 2.0f)) exit(0);
        floatVector results(maxXPartitions + 1);
        HashMap rows = equipartitionYAxis(yVec, yPartitions);
        
        intVector clumps = makeSuperClumps(xVec, rows, (int)(numClumpsFactor * (double)maxXPartitions));
//        for(int i = 0; i< clumps.size(); i++){
//            cout << clumps[i] << " ";
//        }
//        cout << endl;
        
        if(clumps.size() < 2){
            return floatVector(results.size());
        }
        
        int2DVector clumpHistogram = makeClumpHistogram(xVec, yPartitions, rows, clumps);
        
        if(maxXPartitions > clumps.size())
            maxXPartitions = clumps.size();
        
        float2DVector Benefits(clumps.size(), floatVector(maxXPartitions - 1));
        BaseCases(yPartitions, clumps, Benefits, clumpHistogram);
        OptimizeXAxis(maxXPartitions, yPartitions, clumps, Benefits, clumpHistogram);
        
        double yEntropy = Hy(xVec, yPartitions, rows);
        
        for (int xCells = 2; xCells <= maxXPartitions; xCells++)
            results[xCells] = Benefits[clumps.size() - 1][xCells-2] + FLOAT(yEntropy);
        for (int xCells = maxXPartitions + 1; xCells < results.size(); xCells++)
            results[xCells] = results[maxXPartitions];
        
        normalizeResults(yPartitions, results);
        
//        for(int i = 2;  i < results.size(); i++){
//            cout << results[i] << " ";
//        }
//        cout << endl;
        
        
        return results;
        
            
    }
    
private: 
    static HashMap equipartitionYAxis(pointVector &yVec, int yPart){
        HashMap rows;
        intVector yHistogram(yPart);
        pointHashFcn myHashFcn(HASH_SALT);
        int yBinSize = yVec.size()/yPart;
        int currBin, numPlaced;
        currBin = numPlaced = 0;
        
        for(int i = 0; i < yVec.size(); i++){
            int numWithThisY;
            for(numWithThisY = 0; i + numWithThisY < yVec.size() && yVec[i+numWithThisY].Y == yVec[i].Y; numWithThisY++);
            if(yHistogram[currBin] && abs(yHistogram[currBin] + numWithThisY - yBinSize) >= abs(yHistogram[currBin] - yBinSize)){
                currBin++;
                yBinSize = (yVec.size() - numPlaced)/(yPart - currBin);
            }
            for(int j = 0; j < numWithThisY; j++){
                pair<int,int> item(myHashFcn(yVec[i + j]), currBin);
                rows.insert(item);
            }
            yHistogram[currBin] += numWithThisY;
            numPlaced += numWithThisY;
            i += numWithThisY - 1;
                
        }
        return rows;
    }
    
    static intVector makeSuperClumps(pointVector &xVec, HashMap &rows, int numSuperClumps){
        //cout << "(((" << numSuperClumps << ")))" << endl;
        intVector myClumps;
        pointHashFcn myHashFcn(HASH_SALT);
        int desiredSuperClumpSize = xVec.size() / numSuperClumps;
        //cout << "Desired clump size: " << desiredSuperClumpSize << endl;
        myClumps.push_back(-1);
        int i, numInThisSuperClump, numInThisClump;
        i = numInThisSuperClump = 0;
        Point *pXVec_i, *pXVec_ni;
        for(; i < xVec.size(); i += numInThisClump){
            //cout << "Desired clump size: " << desiredSuperClumpSize << endl;
            bool oneXVal, oneRow;
            oneXVal = oneRow = true;
            pXVec_i = &xVec[i];
	    pXVec_ni = &xVec[i];
            for(numInThisClump = 0; i + numInThisClump < xVec.size(); numInThisClump++){
		
                if(rows[myHashFcn(*pXVec_ni)] != rows[myHashFcn(*pXVec_i)])
                    oneRow = false;
                if(pXVec_ni->X != pXVec_i->X){
                    if(!oneRow) break;
                    oneXVal = false;
                }
                if(!oneXVal && rows[myHashFcn(*pXVec_ni)] != rows[myHashFcn(*pXVec_i)])
                    break;
		pXVec_ni++;
            }
            
            if(!oneXVal && i + numInThisClump < xVec.size() && xVec[i + numInThisClump].X == xVec[(i + numInThisClump) - 1].X){
                int j;
                for(j = 1; xVec[(i + numInThisClump) - j].X == xVec[(i + numInThisClump) - 1].X; j++);
                j--;
                
                numInThisClump -= j;
            }
            
            if(numInThisSuperClump && abs((numInThisSuperClump + numInThisClump) - desiredSuperClumpSize) >= abs(numInThisSuperClump - desiredSuperClumpSize)){
                
                myClumps.push_back(i-1);
                numInThisSuperClump = numInThisClump;
                
                if(myClumps.size() - 1 == numSuperClumps)
                    desiredSuperClumpSize = numeric_limits<int>::max();
                else{
                    desiredSuperClumpSize = (xVec.size() - i - 1) / ((numSuperClumps - myClumps.size()) + 1);
                }
            }
            else{
                numInThisSuperClump += numInThisClump;
            }
        }
        myClumps.push_back(xVec.size() - 1);
        myClumps.erase(myClumps.begin());
        //cout << "*" << myClumps.size() << "*" << endl;
        return myClumps;
    }
    
    static int2DVector makeClumpHistogram(pointVector &xVec, int yPart, HashMap &rows, intVector &clumps){
        int2DVector clumpHistogram(clumps.size(), intVector(yPart));
        pointHashFcn myHashFcn(HASH_SALT);
        
        for(int c = 0; c < clumps.size(); c++){
            for(int a = (c != 0) ? clumps[c - 1] + 1 
			                     : 0;
								 a <= clumps[c]; a++)
                clumpHistogram[c][(INT(rows[myHashFcn(xVec[a])]))]++;
        }
        return clumpHistogram;
    }
    
    static void BaseCases(int yPart, intVector &clumps, float2DVector &Benefits, int2DVector &clumpHistogram){
        Benefits[0][0] = (0.0F / 0.0F);
        intVector columnHistogramA(yPart), columnHistogramB(yPart);
		int negMax = -numeric_limits<float>::max();
        for(int c = 1; c < clumps.size(); c++){    
            float maxBenefit = negMax;
            
            
            for(int i = 0; i <= c; i++){
				int *pCHB, *pCH;
				pCHB = &columnHistogramB[0]; pCH = &clumpHistogram[i][0];
                for(int j = 0; j < yPart; j++){
                    //columnHistogramB[j] += clumpHistogram[i][j];
					*pCHB += *pCH;
					pCHB++; pCH++;
				}
            }
            
            for(int lineClumpNum = 0; lineClumpNum < c; lineClumpNum++){
				int *pCHA, *pCHB, *pCH;
				pCHA = &columnHistogramA[0]; pCHB = &columnHistogramB[0]; pCH = &clumpHistogram[lineClumpNum][0];
                for(int j = 0; j < yPart; j++){
                    //columnHistogramB[j] -= clumpHistogram[lineClumpNum][j];
                    //columnHistogramA[j] += clumpHistogram[lineClumpNum][j];
					*pCHB -= *pCH;
					*pCHA += *pCH;
					pCHA++, pCHB++, pCH++;
                }
                
                float benefit = 0.0F;
                int columnA = clumps[lineClumpNum] + 1;
                int columnB = clumps[c] - clumps[lineClumpNum];
                int total = clumps[c] + 1;
				//pCHA = &columnHistogramA[0]; pCHB = &columnHistogramB[0]; pCH = &clumpHistogram[lineClumpNum][0];
                for(int j = 0; j < yPart; j++/*, pCHA++, pCHB++*/){
                    //if(*pCHA){/*columnHistogramA[j] > 0*/
                        //benefit = (benefit + columnHistogramA[j] * log(FLOAT(columnHistogramA[j]) / FLOAT(columnA)));
						//benefit += *pCHA * log(FLOAT(*pCHA)/FLOAT(columnA));
					//}
		//if(*pCHB){/*columnHistogramB[j] > 0*/
                        //benefit = (benefit + columnHistogramB[j] * log((float)columnHistogramB[j] / FLOAT(columnB)));
		//				benefit += *pCHB * log(FLOAT(*pCHB)/FLOAT(columnB));
		//}
		    if(columnHistogramA[j] > 0){
			benefit = (benefit + columnHistogramA[j] * log(FLOAT(columnHistogramA[j]) / FLOAT(columnA)));
		    }
		    if(columnHistogramB[j] > 0){
			benefit = (benefit + columnHistogramB[j] * log(FLOAT(columnHistogramB[j]) / FLOAT(columnB)));
		    }
					
                }

                benefit /= total;
                if(benefit > maxBenefit){
                    maxBenefit = benefit;
                    
                }
            }
            Benefits[c][0] = maxBenefit;
            memset(&columnHistogramA[0], 0, columnHistogramA.size() * sizeof(columnHistogramA[0]));
			memset(&columnHistogramB[0], 0, columnHistogramB.size() * sizeof(columnHistogramB[0]));			
        }  
    }
    
    static void OptimizeXAxis(int xPart, int yPart, intVector &clumps, float2DVector &Benefits, int2DVector &clumpHistogram){
		intVector columnHistogram(yPart);
		float negMax = -numeric_limits<float>::max();
        for(int l = 1; l < xPart - 1; l++){
            for(int c = l + 1; c < clumps.size(); c++){
                float maxBenefit = negMax;
                
                
                for(int lineClumpNum = c; lineClumpNum > l - 1; lineClumpNum--){
					int *pCH_j, *pCH_lcplus1_j;
                    if(lineClumpNum < c){
						pCH_j = &columnHistogram[0]; pCH_lcplus1_j = &clumpHistogram[lineClumpNum + 1][0];
                        for(int j = 0; j < yPart; j++){
                            //columnHistogram[j] += clumpHistogram[lineClumpNum + 1][j];
							*pCH_j += *pCH_lcplus1_j;
							pCH_j++; pCH_lcplus1_j++;
						}
                    }
                    int Ai = clumps[lineClumpNum] + 1;
                    int Bi = clumps[c] - clumps[lineClumpNum];
                    float Hy = 0.0F;
					pCH_j = &columnHistogram[0];
                    for(int j = 0; j < yPart; j++){
                        if(*pCH_j)/*columnHistogram[j]>0*/
                            Hy = (float)(Hy + *pCH_j * log((float)*pCH_j / (float)Bi));
						pCH_j++;
					}

                    float benefit = ((float)Ai / (float)(Ai + Bi)) * Benefits[lineClumpNum][l - 1] + Hy / (float)(Ai + Bi);
                    if(benefit > maxBenefit)
                    {
                        maxBenefit = benefit;
                        
                    }
                }
                
                Benefits[c][l] = maxBenefit;
				memset(&columnHistogram[0], 0, columnHistogram.size() * sizeof(columnHistogram[0]));
            }
        }
    }
    
    static float Hy(pointVector &xVec, int yPart, HashMap &rows){
        pointHashFcn myHashFcn(HASH_SALT);
        float total = 0.0F;
        intVector yHistogram(yPart);
        for(int i = 0; i < xVec.size(); i++)
            yHistogram[INT(rows[myHashFcn(xVec[i])])]++;
        for(int j = 0; j < yPart; j++)
            if(yHistogram[j] > 0)
                total = FLOAT(total + ((float)yHistogram[j] / (float)xVec.size()) * log((double)xVec.size() / (double)yHistogram[j]));
        return total;
    }
    
    static void normalizeResults(int yPart, floatVector &results){
        for(int xPart = 2; xPart < results.size(); xPart++){
            if(xPart > yPart)
                results[xPart] /= log(yPart);
            else
                results[xPart] /= log(xPart);
            results[xPart] = (float)(int)(results[xPart] * 100000.0f) / 100000.0f;
            results[xPart] = min(1.0F, results[xPart]);
        }
    }
    
    
    
};

class approxCharMatrix{
public:
    approxCharMatrix(pointVector &Dataset, float c = .6f, float cf = 15.0f) : regXY(Dataset){
        cellCapExp = c; clumpFactor = cf;
        //pointVector regXY(Dataset.begin(), Dataset.end());
        sort(regXY.begin(), regXY.end(), pointXCompare());
//        for(int i = 0; i < Dataset.size(); i++){
//            cout << regXY[i].X << ", " << regXY[i].Y << endl;
//        }
        pointVector regYX(Dataset.begin(), Dataset.end());
        sort(regYX.begin(), regYX.end(), pointYCompare());
        pointVector transYX(Dataset.size());
        pointVector transXY(Dataset.size());
        transform(regXY.begin(), regXY.end(), transYX.begin(), swapXY);
        transform(regYX.begin(), regYX.end(), transXY.begin(), swapXY);        
        
        cellCap = INT(pow(regXY.size(), cellCapExp));
        cellCap = max(cellCap, 10);
        
        scores.resize(cellCap/2 + 1);
        
        
//        cout << endl << "Analyzing the data set: " << endl
//             << "X-range: " << regXY[0].X << "-" << regXY[regXY.size()-1].X << endl
//             << "Y-range: " << regYX[0].Y << "-" << regYX[regYX.size()-1].Y << endl
//             << "# common values: " << regXY.size() << endl
//             << "Max # of cells allowed: " << cellCap << endl;
        
        //threadPacket *test = new threadPacket(transYX, transXY, cellCap / xCells);
        
        MICHeuristic myCalc;
        floatVector results(cellCap, 0.0f);
       
        
        for(int xCells = 2; xCells <= cellCap/2; xCells++){
            scores[xCells].resize(cellCap/xCells+1, 0.0f);
            
            
            results = myCalc.approxMaxMI(transXY, transYX, cellCap / xCells, xCells, clumpFactor);
            for(int yCells = 2; yCells < scores[xCells].size(); yCells++){
                scores[xCells][yCells] = results[yCells];
            }
        }
        
        for(int yCells = 2; yCells <= cellCap/2; yCells++){            
            results = myCalc.approxMaxMI(regXY, regYX, cellCap / yCells, yCells, clumpFactor);
            for(int xCells = 2; xCells < scores[yCells].size(); xCells++){
                scores[xCells][yCells] = max(scores[xCells][yCells], results[xCells]);
            }
        }
        
       
    }
    float MIC(Grid &optGrid){
        float MIC = -numeric_limits<float>::max();
        int maxX, maxY;
        maxX = maxY = -1;
        //cout << scores[2][2] << endl;
        for(int x = 2; x < scores.size(); x++){
            for(int y = 2; y < scores[x].size(); y++){
                //cout << scores[y][x] << ", ";
                if((scores[x][y] <= MIC) &&
                  ((scores[x][y] != MIC) || (x * y >= maxX *maxY))){ // iqnore grid if 
                    continue;
                }
                MIC = scores[x][y];
                
                maxX = x;
                maxY = y;
            }
            //cout << endl;
            
        }
        optGrid.X = maxX;
        optGrid.Y = maxY;
        return MIC; 
        //return scores[2][2];
    }
private:
    
    float cellCapExp; //Params
    double clumpFactor;//""
    int cellCap;
    float2DVector scores;
    pointVector &regXY;
};

class Utility{
public:
    double randNorm(double sigma, int range, double mu=0.0){
        static bool deviateAvailable=false;
        static float storedDeviate;
        double polar, rsquared, var1, var2;
        if (!deviateAvailable){
            do{
                var1=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
                var2=2.0*( double(rand())/double(RAND_MAX) ) - 1.0;
                rsquared=var1*var1+var2*var2;
            }while ( rsquared>=1.0 || rsquared == 0.0);
            polar=sqrt(-2.0*log(rsquared)/rsquared);	
            storedDeviate=var1*polar;
            deviateAvailable=true;	
            return (var2*polar*sigma + mu);
        }
        else{
            deviateAvailable=false;
            return (storedDeviate*sigma + mu);
        }
    }
    
    pointVector assembleData(int numPoints, int choice, double stdev, double noise = 0, int range = 1000){
        timeval t1;
        gettimeofday(&t1,0);
        srand(t1.tv_sec*1000000LL + t1.tv_usec);
        pointVector Points;
        int ymax = INT(numPoints*noise);
        for(int y = 0; y < ymax; y++){//only activated if noise > 0 and INT(numPoints*noise) > 0
            Point myPoint;
            myPoint.X = rand()%range; myPoint.Y = rand()%range;
            Points.push_back(myPoint);
        }
            
        for(int y = 0; y < numPoints; y++){
            Point myPoint;
            myPoint.X = rand()%range;
            myPoint.Y = myFcns(choice, myPoint, range, stdev);
            Points.push_back(myPoint);
        }
        
        return Points;
    }
    
    void makeCSV(pointVector Points){
        ofstream myCSV;
        myCSV.open("Points.csv");
        myCSV << "X,Y\n"; 
        for(int cursor = 0; cursor < Points.size(); cursor++){
            myCSV << Points[cursor].X << "," << Points[cursor].Y << "\n";
        }
        myCSV.close();
    }
        
    int myFcns(int choice, Point myPoint, int range, double sigma){
        switch(choice){
            case 0:        
                return rand()%range;
            case 1:
                return myPoint.X+INT(randNorm(sigma,range));
            case 2:
                return log(myPoint.X)*range+INT(randNorm(sigma,range));
            case 3:
                return sin(myPoint.X)*range+INT(randNorm(sigma,range));
            case 4:
                return sin(log(myPoint.X))*range+INT(randNorm(sigma,range));
            default:
                return rand()%range;

        }
    }
    pointVector readInFromCSV(){
        string strPoint;
        ifstream inCSV;
        pointVector Points;
        inCSV.open("Points.csv");
        getline(inCSV, strPoint);
        while(getline(inCSV, strPoint)){
            Point myPoint;
            int comma = strPoint.find(',');
            myPoint.X = (atof(strPoint.substr(0, comma).c_str())*1000.0f);
            myPoint.Y = (atof(strPoint.substr(comma+1, strPoint.size()-1).c_str())*1000.0f);
            Points.push_back(myPoint);
            //cout << myPoint.X << ", " << myPoint.Y << endl;
        }
        
        inCSV.close();
        return Points;
        
    }
    
    stringVector readToInt2DVecFromCSV(float2DVector &Vars, string &filename){
        
        // Save for later; generalized variable caching
        ifstream inCSV;
        inCSV.open(filename.c_str());
        int comma = 0; int begin = 0;
        stringVector names;

        string line;
        
        getline(inCSV, line);
        //strip first
        comma = line.find(',');
        line = line.substr(comma+1, line.size());
        do{
            comma = line.find(',');
            names.push_back(line.substr(0, comma));
            line = line.substr(comma+1, line.size());
            
            
            //begin = comma + 1;   
        }while(comma != -1);
        names[names.size()-1] = names[names.size()-1].substr(0, line.size()-1);
        getline(inCSV, line);
        comma = line.find(',');
        line = line.substr(comma+1, line.size());
        do{
            comma = line.find(',');
            
            
            floatVector temp;
            temp.push_back(atof(line.substr(0, comma).c_str())*1000.0f);
            Vars.push_back(temp);
            line = line.substr(comma+1, line.size());
            
            
            //begin = comma + 1;   
        }while(comma != -1);

        
        int lineNo = 1;
        while(getline(inCSV, line)){
	    comma = line.find(',');
	    line = line.substr(comma+1, line.size());
            for(int i = 0, begin = 0; i < Vars.size(); i++){
                comma = line.find(',');
				
                Vars[i].push_back(atof(line.substr(0, comma).c_str())*1000.0f);
                line = line.substr(comma+1, line.size());
				
                //begin = comma + 1;
            }
            lineNo++;
        }
        inCSV.close();
	
        return names;
    }
    
    pointVector constructPoints(floatVector one, floatVector two){
        pointVector myPoints(one.size());
        for(int i = 0; i < one.size(); i++){
            Point temp;
            temp.X = one[i]; temp.Y = two[i];
            myPoints[i] = temp;
        }
        return myPoints;
    }
};

class pairProcessor{ // For future use...
public:
    pairProcessor(approxCharMatrix Analysis, string var1, string var2){
        this->var1 = var1, this->var2 = var2;
        
        this->myAttributes[MIC]  = Analysis.MIC(this->optimalGrid);
        this->myAttributes[PRIM] = this->optimalGrid.X;
        this->myAttributes[SEC]  = this->optimalGrid.Y;
        this->myAttributes[MAS]  = 1.337;
        this->myAttributes[MEV]  = 1.337;
        this->myAttributes[MCN]  = 1.337;
        
    }
    
    void printReport(){
        
        ofstream outCSV;
        string filename = var1+","+var2+".o.csv";
        
        if     (myAttributes[MIC] >  0.99) filename = "5-"     + filename;
        else if(myAttributes[MIC] >  0.79) filename = "4-"     + filename;
        else if(myAttributes[MIC] >  0.59) filename = "3-"     + filename;
        else if(myAttributes[MIC] >  0.39) filename = "2-"     + filename;
        else if(myAttributes[MIC] >  0.19) filename = "1-"     + filename;
        else                               filename = "0-"     + filename;
        
        cout << "Printing " << filename << "..." << endl;
        outCSV.open(filename.c_str()); 
        const string outNames[END] = {"MIC", "PRIM", "SEC", "MAS", "MEV", "MCN"};
        for(int i = 0; i < END; i++){
            outCSV << outNames[i] << ",";
        }
        outCSV << "\n";
        for(int i = 0; i < END; i++){
            outCSV << myAttributes[i] << ",";
        }
        outCSV << "\n";
        outCSV.close();
    }
    
private:
    enum chars{
        MIC,  // Maximal Information Coefficient; a.k.a. "Level of Order".
        PRIM, // Optimal Grid Resolution Upon Primary Axis.
        SEC,  // Optimal Grid Resolution Upon Secondary Axis.
        MAS,  // Maximum Asymmetry Score; a.k.a. "Non-Monotonicity".
        MEV,  // Maximal Edge Value; a.k.a. "Closeness to Being a Function".
        MCN,  // Minimum Cell Number; a.k.a. "Complexity".
        END   // Number of Enums. Be Smart--this may not work if you fiddle with
              // enum values, and will likely break other parts.
    };
    
    string var1, var2;
    float myAttributes[END];
    Grid optimalGrid;
    
};

class threadPacket{
    
public:
    threadPacket(float2DVector &data, float2DVector &origResults, int core,
            int start, int end) : results(origResults), mySet(data){
        
        this->Core  = core;
        this->start = start;
        this->end   = end;
        //this->results = origResults;
    }
    
    ~threadPacket(){
    }
    float2DVector &mySet;
    float2DVector &results;
    int start, end;
    
    int Core;
};
void *runThread(void *data){
    Utility myUtil;
    threadPacket *myPack = static_cast<threadPacket *>(data);
    for(int first = myPack->start; first < myPack->end; first++){
        int i;
        for(i = 0; i < first+1; i++){
            //pad results
            myPack->results[first].push_back(-1.0f);
            
            
        }
		//if(myPack->Core == 47) cout << myPack->mySet.size() << endl;
        for(int second = first + 1; second < myPack->mySet.size(); second++){
            //if(myPack->Core == 47) cout << first << ", " << second << endl;
            pointVector temp = myUtil.constructPoints(myPack->mySet.at(first), myPack->mySet.at(second));
            approxCharMatrix acm(temp);
            Grid optgrid;
            float score = acm.MIC(optgrid);
            myPack->results[first].push_back(score);
			//if(myPack->Core == 47) cout << first << " " << second << " " <<  endl;
            
        }
        
    }
    cout << "Thread " << myPack->Core << " has finished." << endl;
    sem_post(&semDuo[myPack->Core]);

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    timeval t0,t1;
    gettimeofday(&t0,0);
    Utility myUtil;
    stringVector headers;
    float2DVector variables;
    int nproc = (argc > 1) ? atoi(argv[1]) : 1;
    int npair = (argc > 2) ? atoi(argv[2]) : 2;
    string filename;// = "gene_sel_190.csv";
    if(argc < 4){
        cout << "No input file! Exiting."<< endl;
        return 1;
    }
    else{
        filename = argv[3];
    }
    semDuo = (sem_t*)malloc(nproc*sizeof(sem_t));    
    
    float size = (float)npair/(float)nproc;
    
    double sigma = 0;
    for(int i = 0; i < npair; i++){
        sigma+=i;
    }
    
    int slice = sigma / (nproc);
    
    
    int marker = 0;
    int indices[nproc];
    memset(&indices[0], 0, nproc * sizeof(int));
    int index = 0;
    int p;
    for(p = 1; p < npair; p++){
        if((marker + (npair-p)) < slice){
	    marker += (npair - p);
	    
	}
	else{
	    indices[index++] = p;    
	    marker = 0;
	    slice = sigma/(nproc-index);
	}
	sigma -= (npair - p);
	
    }
    indices[index] = npair;
    int j;
    //indices[41] = npair;
    for(j = 0; j < nproc; j++){
        
	if(indices[j] == 0)
	  break;
    }
    nproc = j;
    //exit(0);
    
    ofstream outCSV;
    outCSV.open("resultsForGeneAnalysis.csv");
    headers = myUtil.readToInt2DVecFromCSV(variables, filename);
    //headers[headers.size()-1] = headers[headers.size()-1].substr(0, headers[headers.size()-1].size()-2);
    float2DVector results(headers.size(), floatVector());
	
    pthread_t threadDuo[nproc];
    pthread_attr_t tattr;
    for(int i = 0; i < nproc; i++){
        sem_init(&semDuo[i],0,1);
        sem_wait(&semDuo[i]);
    }
    pthread_attr_init(&tattr);
    pthread_attr_setdetachstate(&tattr, PTHREAD_CREATE_DETACHED);
    vector<threadPacket*> threads;
    int i;
	gettimeofday(&t1,0);
	//cout << (long int)(t1.tv_sec - t0.tv_sec) << "." << abs((long int)(t1.tv_usec - t0.tv_usec)) << " seconds to set up for threads." << endl;
    gettimeofday(&t0,0);
	int start = 0; int end;
	int edge = 0;
	for(i = 0; i < nproc; i++){
	  
	  threads.push_back(new threadPacket(variables, results, i, edge, indices[i]));
	  //cout << i*size << " " << (i+1)*size << endl;
	  pthread_create(&threadDuo[i], &tattr, &runThread, static_cast<void *>(threads[i]));
	  edge = indices[i];
	}
    //threads[i] = new threadPacket(variables, results, i, round(i*size), results.size());
    //pthread_create(&threadDuo[i], &tattr, &runThread, static_cast<void *>(threads[i]));
    //cout << i*size << " " << results.size() << endl;
    //exit(0);
    
    for(int i = 0; i < nproc; i++){
        sem_wait(&semDuo[i]);
        delete threads[i];
    }
    cout << "done" << endl;
    free(semDuo);
    //outCSV << ",";
    //for(int i = 0; i < headers.size(); i++){
    //    outCSV << headers[i];
	//	if (i+1 < headers.size()) outCSV << ",";
    //}
    //outCSV << "\n";
    for (int i = 0; i < results.size(); i++){
      for(int j = 0; j < results.size(); j++){
	if(results[i][j] >= 0){ outCSV << headers[i] << "," << headers[j] << "," << results[i][j] << "\n";}
      }
    }
//    for(int i = 0; i < results.size(); i++){
//        outCSV << headers[i] << ",";
//        for(int j = 0; j < results[i].size(); j++){
//            if(results[i][j] >= 0) outCSV << results[i][j];
//            outCSV << ",";
//			cout << "*" << results[i][j] << "*" << endl;
//        }
//        outCSV << "\n";
//    }

    gettimeofday(&t1,0);
    cout << (long int)(t1.tv_sec - t0.tv_sec) << "." << abs((long int)(t1.tv_usec - t0.tv_usec)) << " seconds to run all threads."  << endl;
    outCSV.close();
    

    return 0;
}



