/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
//
// Copyright (c) 2006 Georgia Tech Research Corporation
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation;
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Author: George F. Riley<riley@ece.gatech.edu>
//

// ns3 - On/Off Data Source Application class
// George F. Riley, Georgia Tech, Spring 2007
// Adapted from ApplicationOnOff in GTNetS.

#ifndef __EnergyEfficient_application_h__
#define __EnergyEfficient_application_h__

#include "ns3/address.h"
#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/myBeaconing-wifi-mac.h"
#include "ns3/traced-callback.h"
#include "ns3/callback.h"
#include "ns3/global-value.h"
#include <vector>
#include "ns3/event-id.h"
#include "ns3/data-rate.h"

namespace ns3 {

class Address;
class RandomVariable;
class Socket;
//class MyBeaconingWifiMac;	

/**
 * \ingroup applications 
 * \defgroup EnergyEfficientApplication
 *
 * Creates a EnergyEfficient tree topology
 * Author: Patricia Ruiz
**/

class EnergyEfficientApplication : public Application 
	
{
//typedef sgi::hash_map<Ipv4Address, Address, Ipv4AddressHash> NeighborHashMap;
//typedef sgi::hash_map<Ipv4Address, Address, Ipv4AddressHash>::iterator NeighborHashMapIterator;	
	typedef std::vector<GlobalValue *> VectorA;

public:
	
  static TypeId GetTypeId (void);
  EnergyEfficientApplication ();
  virtual ~EnergyEfficientApplication();
	  
protected:
  virtual void DoDispose (void);
  void SendBroadcast();
	void ReceiveData (Ptr<Socket> socket);	
private:
	virtual void StartApplication (void);    // Called at time specified by Start
	void CheckForwarding (uint32_t idMessage);
	void SetRandomDelay(uint32_t idMessage);
	void ReSendBroadcast(uint32_t idMessage);
	double CalculateNeighborsPower();
	void Send (Ptr<Packet> message);
	static void NeighborLost (std::string context, Ptr<const Packet> packet, Mac48Address addr);
	static void NewNeighborFound (std::string context, Ptr<const Packet> packet, Mac48Address addr, double rxPwDbm);

	TypeId          m_tid;
	double          m_pwRxDbm;
    Ptr<Socket>     m_EnergyEfficientSocket; // Associated socket
	Ipv4Address     m_addr; 	
	Time            m_startingBroadcast; //Starting broadcast
	double			m_randomInterval;// delay before forwarding
	double          m_energyThreshold;
	bool            m_forwardingNode;
	DataRate        m_cbrRate;      // Rate that data is generated
	EventId         m_eventId;
	Mac48Address    m_macAddr;      // Size of packets    

	typedef	std::map <Mac48Address,double> MacAddrMap;
	typedef   std::map<Mac48Address, double>::iterator MacAddrMapIterator;	
	MacAddrMap    m_neighborList;
	
	typedef	std::map <uint32_t,Ptr<Packet> > MessagesMap;
	typedef   std::map<uint32_t,Ptr<Packet> >::iterator MessagesMapIterator;	
	MessagesMap    m_messagesList;
	typedef VectorA::const_iterator Iterator;

	TracedCallback<Ptr<const Packet> > m_txTrace;
	TracedCallback<Ptr<const Packet> >m_rxTrace;
	

	

	
};
	

} // namespace ns3

#endif

