/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006,2007 INRIA
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#include "ns3/object.h"
#include "ns3/ssid.h"
#include <iostream>
#include <fstream>
#include "ns3/inet-socket-address.h"
#include "ns3/beaconing-application.h"
#include "ns3/dca-txop.h"
#include "ns3/global-value.h"
#include "ns3/random-variable.h"
#include <vector>
#include <sys/time.h>
#include <time.h>
#include "ns3/vector.h"
#include "ns3/yans-wifi-helper.h"




namespace ns3{


class ns3DFCNOptimizedCall{

public:
	typedef std::vector<GlobalValue *> VectorA;
	typedef VectorA::const_iterator Iterator;
	//static TypeId GetTypeId (void);
	ns3DFCNOptimizedCall ();
	//~ns3DFCNOptimizedCall ();
	double* RunExperimentDFCNOptimized (int numberDevices, int j,double minDelay, double maxDelay, double minBenefit, double densityThreshold,double safeDensity);//,char* name);//double &energy_used, int &total_coverage, double &broadcast_time);//, double *energy, double *forward );
    void Run (NetDeviceContainer devices,YansWifiPhyHelper wifiPhy,NodeContainer c, double minDelay, double maxDelay, double minBenefit, double densityThreshold,double safeDensity);
	void CreateSimulation (ns3DFCNOptimizedCall exp, int numberDevices, double minDelay, double maxDelay, double minBenefit, double densityThreshold,double safeDensity);
	void SetToZero (Iterator i);
	void LogToFile (Iterator i,char * name);
	void ObtainObjectives (Iterator i,double &energy_used, double &total_coverage, double &broadcast_time);
	
private:
	
    uint32_t m_bytesTotal;

};
}

