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

#ifndef __beaconing_application_h__
#define __beaconing_application_h__

#include "ns3/address.h"
#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/ptr.h"
#include "ns3/data-rate.h"
#include "ns3/random-variable.h"
#include "ns3/traced-callback.h"
#include "ns3/sgi-hashmap.h"
#include "ns3/DAGRS-application.h"
//#include <asm/atomic.h>

namespace ns3 {

class Address;
class RandomVariable;
class Socket;

/**
 * \ingroup applications 
 * \defgroup onoff OnOffApplication
 *
 * This traffic generator follows an On/Off pattern: after 
 * Application::StartApplication
 * is called, "On" and "Off" states alternate. The duration of each of
 * these states is determined with the onTime and the offTime random
 * variables. During the "Off" state, no traffic is generated.
 * During the "On" state, cbr traffic is generated. This cbr traffic is
 * characterized by the specified "data rate" and "packet size".
 */
 /**
 * \ingroup onoff
 *
 * \brief Generate traffic to a single destination according to an
 *        OnOff pattern.
 *
 * This traffic generator follows an On/Off pattern: after 
 * Application::StartApplication
 * is called, "On" and "Off" states alternate. The duration of each of
 * these states is determined with the onTime and the offTime random
 * variables. During the "Off" state, no traffic is generated.
 * During the "On" state, cbr traffic is generated. This cbr traffic is
 * characterized by the specified "data rate" and "packet size".
 *
 * Note:  When an application is started, the first packet transmission
 * occurs _after_ a delay equal to (packet size/bit rate).  Note also,
 * when an application transitions into an off state in between packet
 * transmissions, the remaining time until when the next transmission
 * would have occurred is cached and is used when the application starts
 * up again.  Example:  packet size = 1000 bits, bit rate = 500 bits/sec.
 * If the application is started at time 3 seconds, the first packet
 * transmission will be scheduled for time 5 seconds (3 + 1000/500)
 * and subsequent transmissions at 2 second intervals.  If the above
 * application were instead stopped at time 4 seconds, and restarted at
 * time 5.5 seconds, then the first packet would be sent at time 6.5 seconds,
 * because when it was stopped at 4 seconds, there was only 1 second remaining
 * until the originally scheduled transmission, and this time remaining
 * information is cached and used to schedule the next transmission
 * upon restarting.
 */
	
//class MyCb {
//public:
//	int LossingNeighbor (Ipv4Address neighbor);
//};
	
class BeaconingApplication : public Application 
{
	
	
public:
  static TypeId GetTypeId (void);

  BeaconingApplication ();

  virtual ~BeaconingApplication();
  
	DAGRSApplication da;

	
protected:
  virtual void DoDispose (void);
private:
	
	typedef sgi::hash_map<Ipv4Address, uint16_t, Ipv4AddressHash> NeighborHashMap;
	typedef sgi::hash_map<Ipv4Address, uint16_t, Ipv4AddressHash>::iterator NeighborHashMapIterator;	
	
  // inherited from Application base class.
  virtual void StartApplication (void);    // Called at time specified by Start
//  virtual void StopApplication (void);     // Called at time specified by Stop

//  //helpers
//  void CancelEvents ();
//
//  void Construct (Ptr<Node> n,
//                  const Address &remote,
//                  std::string tid,
//                  const RandomVariable& ontime,
//                  const RandomVariable& offtime,
//                  uint32_t size);
//
//
//  // Event handlers
 // void StartSending();
//  void StopSending();
  void SendPacket();
	void NeighborLost (Ipv4Address neighbor);
	void ReceiveBeacon (Ptr<Socket> socket);	
//	void ListenApplication(); 
	void Calling (Ipv4Address neighbor);
//	Ptr<Packet> BeaconingApplication::Recv (uint32_t maxSize, uint32_t flags);
//
  Ptr<Socket>     m_socket;       // Associated socket
  Ptr<Socket>     m_socketListen;       // Associated socket
  Ipv4Address     m_addr;   // Own addr
  uint16_t        m_numberBeaconLost;	
  uint32_t        m_nodeId;	
  Time            m_IntervalTime;      // beacon interval time Time
  DataRate        m_cbrRate;      // Rate that data is generated
  uint32_t        m_pktSize;      // Size of packets
  uint32_t        m_totBytes;     // Total bytes sent so far
  TypeId          m_tid;
	NeighborHashMap m_neighborList;
	NeighborHashMapIterator m_neighborListIterator; 
	
	TracedCallback<Ptr<const Packet> > m_txTrace;
	TracedCallback<Ptr<const Packet> >m_rxTrace;
	

//  
private:
  void ScheduleNextTx();
	Ptr<Packet> CreatePacket (Ptr<Node> dev);

//  void ScheduleStartEvent();
//  void ScheduleStopEvent();
//  void ConnectionSucceeded(Ptr<Socket>);
//  void ConnectionFailed(Ptr<Socket>);
//  void Ignore(Ptr<Socket>);
	
	

	
};
	
//class BeaconingMessage:BeaconHeader{ 
//	
//	//   HELLO Message Format
//	//
//	//        0               1               2               3
//	//        0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 
//	//
//	//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
//	//       |     Header     |        IdDevice               |
//	//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
//	//       |         Neighbor Interface Address             |
//	//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
//	//       |         Neighbor Interface Address             |
//	//       +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
//	//    (etc.)	
//	
//public:
//	Ptr<Packet> CreatePacket (Ptr<Node> dev);
////		static TypeId
////	BeaconingMessage::GetTypeId (void);
////	
////	BeaconingMessage ();
////	
////	virtual ~BeaconingMessage();
//	
//	//void SetMaxBytes(uint32_t maxBytes);
//	
//protected:
//	// virtual void DoDispose (void);	
//private:
//	BeaconHeader header;
//	uint32_t nodeId;
//	Address nodeAddress;
//	//TracedCallback <Ptr<const Packet> > m_Trace;
//	BeaconHeader GetHeader();
//	
//	uint32_t GetNodeId ();
//	void SetNodeId (uint32_t nodeId);
//	void SetAddress (Address addr);
//	Address GetAddress();
//	
//}; //end Beaconing Messages
} // namespace ns3

#endif

