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

#ifndef __DAGRS_application_h__
#define __DAGRS_application_h__

#include "ns3/address.h"
#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/data-rate.h"
#include "ns3/random-variable.h"
#include "ns3/traced-callback.h"
#include "ns3/sgi-hashmap.h"


namespace ns3 {

class Address;
class RandomVariable;
class Socket;

/**
 * \ingroup applications 
 * \defgroup DAGRSApplication
 *
 * Creates a DAGRS tree topology
 * Author: Patricia Ruiz
**/



class DAGRSApplication : public Application 
	
{
typedef sgi::hash_map<Ipv4Address, Address, Ipv4AddressHash> NeighborHashMap;
typedef sgi::hash_map<Ipv4Address, Address, Ipv4AddressHash>::iterator NeighborHashMapIterator;	

public:
	
  int NewNeighbor (Ipv4Address neighbor, Address m_macAdd);
  int LossingNeighbor (Ipv4Address neighbor);
  static TypeId GetTypeId (void);
  DAGRSApplication ();
  virtual ~DAGRSApplication();
  		
protected:
  virtual void DoDispose (void);
private:
    Ptr<Packet> CreateStartMerge ();
	Ptr<Packet> CreateAckMerge ();
    Ptr<Packet> CreateIamToken ();
	Ptr<Packet> CreateMovingToken ();
	Address RandomTraversalStrategy();
	void SendToken();
	void UpdateNeighborhood ();
	void AnyIsTokenElapsed();
	void StartMergerElapsed();
	virtual void StartApplication (void);    // Called at time specified by Start
	void ReceiveTokenAnswer (Ptr<Socket> socket);
	void Merge(Ipv4Address ip4, Address mac);
  // inherited from Application base class.

	NeighborHashMap m_oneHopNeighborList;
	NeighborHashMap m_treeNeighborList;
	bool m_ackMerger;
	bool m_token;
	bool m_MergeDone;
	bool m_doingMerging;
	bool m_changeInNeighborhood;
    Ptr<Socket>     m_dagrsSocket; // Associated socket
	Ptr<Socket>     m_socketListen;       // Associated socket
	Ipv4Address     m_addr; 
	Address         m_neighborTokenSent;
//	Address         m_macAdd;// Own addr
//	uint16_t        m_numberBeaconLost;	
//	uint32_t        m_nodeId;	
	Time            m_waitingAnyToken;      // beacon interval time Time
//	DataRate        m_cbrRate;      // Rate that data is generated
//	uint32_t        m_pktSize;      // Size of packets
//	uint32_t        m_totBytes;     // Total bytes sent so far
	TypeId          m_tid;


//	NeighborHashMapIterator m_neighborListIterator; 
//	
	TracedCallback<Ptr<const Packet> > m_txTrace;
	TracedCallback<Ptr<const Packet> >m_rxTrace;
	

	

	
};
	

} // namespace ns3

#endif

