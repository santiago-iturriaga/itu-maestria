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
#include "ns3/DAGRS-header.h"
#include "ns3/greenDBDA-helper.h"
#include "ns3/global-value.h"
#include "ns3/random-variable.h"
#include <vector>
#include <sys/time.h>
#include <time.h>
#include "ns3/vector.h"
#include "ns3/yans-wifi-helper.h"




using namespace ns3;

class ns3AEDBRestrictedCall
{
	typedef std::vector<GlobalValue *> VectorA;
	typedef VectorA::const_iterator Iterator;
public:
	ns3AEDBRestrictedCall ();
	//ExperimentJNA (std::string name);
	double* RunExperimentAEDBRestricted (int numberDevices, int j,double minDelay, double maxDelay, double borderThreshold, double marginThreshold,int neighborsThreshold);//,char* name);//double &energy_used, int &total_coverage, double &broadcast_time);//, double *energy, double *forward );

	//void SetPosition (Ptr<Node> node, Vector position);
	//Vector GetPosition (Ptr<Node> node);
	//void AdvancePosition (Ptr<Node> node);
    void Run (NetDeviceContainer devices,YansWifiPhyHelper wifiPhy,NodeContainer c, double minDelay, double maxDelay, double borderThreshold, double marginThreshold,int neighborsThreshold);
	//	 void ReceivePacket (Ptr<Socket> socket);
	void CreateSimulation (ns3AEDBRestrictedCall exp, int numberDevices, double minDelay, double maxDelay, double borderThreshold, double marginThreshold,int neighborsThreshold);
	void SetToZero (Iterator i);
	void LogToFile (Iterator i,char * name);
	void ObtainObjectives (Iterator i,double &energy_used, double &total_coverage,double &total_forwarding,  double &broadcast_time);
	
private:
	
	//static void CourseChange (std::string foo, Ptr<const MobilityModel> mobility);
    uint32_t m_bytesTotal;
   // Gnuplot2dDataset m_output;

};


