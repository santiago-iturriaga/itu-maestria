/*
 *  JNATest.c
 *  
 *
 *  Created by Patricia on 17/05/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */
#include "JNATest.h"
#include <iostream>
#include "ns3/ns3AEDBRestrictedCall.h"

int main (int argc, char* argv[]) {
	
	//std::cout<<"Running Experiment class in ns3"<<std::endl;
	ns3AEDBRestrictedCall exp = ns3AEDBRestrictedCall();
	int numberDevices = atoi(argv[1]); // 25 for 100 devices/km^2 density, 50 for 200, and 75 for 300
	int j = atoi(argv[2]);  // number of independent runs of the simulator, if > 1, the results given are averaged over all the runs
	double minDelay = atof(argv[3]); //  Be sure that minDelay<maxdelay
	double maxDelay = atof(argv[4]);
	double borderThreshold = atof(argv[5]);
	double marginThreshold = atof(argv[6]);
        int    neighborThreshold = atoi(argv[7]);
	double energy_used = 0.0;
	int total_coverage = 0;
	double broadcast_time = 0.0;

    if (borderThreshold < 0) borderThreshold = borderThreshold * -1;

    double* aux;
    aux = (double*)malloc(4*sizeof(double));
    //printf("%f %f %f", aux[0], aux[1], aux[2]);
    // Call the ns3 function, and the results for the three objectives and the time (which is used as a constraint) are put in aux
	aux = exp.RunExperimentAEDBRestricted (numberDevices, j, minDelay, maxDelay, borderThreshold, marginThreshold, neighborThreshold);
  
    printf("Energy, coverage, forwardings, and time\n");
    printf("%f %f %f %f\n", aux[0], aux[1], aux[2], aux[3]);

return 0;
}	

