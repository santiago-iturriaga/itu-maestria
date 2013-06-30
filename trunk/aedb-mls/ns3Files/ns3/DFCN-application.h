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

#ifndef __DFCN_application_h__
#define __DFCN_application_h__

#include "ns3/address.h"
#include "ns3/application.h"
#include "ns3/event-id.h"
#include "ns3/myBeaconing-wifi-mac.h"
#include "ns3/traced-callback.h"
#include "ns3/callback.h"
#include "ns3/global-value.h"
#include <vector>
#include <list>
#include "ns3/event-id.h"
#include "ns3/data-rate.h"

namespace ns3 {

class Address;
class RandomVariable;
class Socket;
//class MyBeaconingWifiMac;	

/**
 * \ingroup applications 
 * \defgroup DFCNApplication
 *
 * Creates a NewEnergyEfficientAdaptiveOptimized tree topology
 * Author: Patricia Ruiz
**/

class DFCNApplication : public Application 
	
{	
	// Vector with all the global variables used for statistics
	typedef std::vector<GlobalValue *> VectorA;
	typedef VectorA::const_iterator Iterator;

	// List of neighbor addresses for the DFCN message header
	typedef std::list<Mac48Address> ListAddress;
	typedef std::list<Mac48Address>::iterator ListAddressIterator;
	ListAddress             m_neighborList;
	ListAddress             m_messageNeighborList;
	
public:
	
  static TypeId GetTypeId (void);
  DFCNApplication ();
  virtual ~DFCNApplication();
	  
protected:
  virtual void DoDispose (void);
  void SendBroadcast();
	void ReceiveData (Ptr<Socket> socket);	
private:
	virtual void StartApplication (void);    // Called at time specified by Start
	void CheckForwarding (uint32_t idMessage);
	void SetRandomDelay(uint32_t idMessage);
	void ReSendBroadcast(uint32_t idMessage);
	void Send (Ptr<Packet> message);
	static void NeighborLost (std::string context, Ptr<const Packet> packet, Mac48Address addr);
	static void NewNeighborFound (std::string context, Ptr<const Packet> packet, Mac48Address addr, double rxPwDbm);
	Mac48Address GetMacAddress(Ipv4Address multicastGroup);
	void CancelDelayAndSendAllMessages();
	void CalculateBenefit(uint32_t messageId, ListAddress messageList);
	
	TypeId          m_tid;
	double          m_pwRxDbm;
    Ptr<Socket>     m_DFCNSocket; // Associated socket
	Ipv4Address     m_addr; 	
	Time            m_startingBroadcast; //Starting broadcast
	double			m_randomIntervalMin;// delay before forwarding
	double			m_randomIntervalMax;// delay before forwarding
	double          m_minBenefit;
	double          m_benefitThreshold;
	double          m_densityThreshold;
	double          m_safeDensity;
	bool            m_doingRandom;
	DataRate        m_cbrRate;      // Rate that data is generated
	EventId         m_eventId;
	Mac48Address    m_macAddr;      // Size of packets  
	Mac48Address    m_neighborMac48Addr;
	Address         m_neighborAddress;
	uint32_t        m_messageID;

	// Map cointaining the neighbors address and reception power
	typedef	std::map <Mac48Address,double> MacAddrMap;
	typedef   std::map<Mac48Address, double>::iterator MacAddrMapIterator;	
	MacAddrMap              m_neighborMap;	

	// Map with Id of teh receoved message and the message itself
	typedef	std::map <uint32_t,Ptr<Packet> > MessagesMap;
	typedef   std::map<uint32_t,Ptr<Packet> >::iterator MessagesMapIterator;	
	MessagesMap    m_messagesList;
	

	TracedCallback<Ptr<const Packet> > m_txTrace;
	TracedCallback<Ptr<const Packet> >m_rxTrace;
	

	

	
};
	

} // namespace ns3

#endif

