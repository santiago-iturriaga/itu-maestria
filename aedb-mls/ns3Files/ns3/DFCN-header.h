/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005 INRIA
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

#ifndef DFCN_HEADER_H
#define DFCN_HEADER_H

#include <stdint.h>
#include <string>
#include "ns3/header.h"
#include "ns3/ipv4-address.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/mac48-address.h"
#include <list>


using namespace std;

namespace ns3 {
/**
 * \ingroup udp
 * \brief Packet header for UDP packets
 *
 * This class has fields corresponding to those in a network UDP header
 * (port numbers, payload size, checksum) as well as methods for serialization
 * to and deserialization from a byte buffer.
 */
	
	
	/**********
	 __________________________
	 | MessageID | Lenght list  |
	 |__________________________|
	 |           List	        |
	 |__________________________|
	 *********/	
	
	
class DFCNHeader: public Header 
{
	
public:
	typedef list<Mac48Address> ListAddress;
	typedef list<Mac48Address>::iterator ListAddressIterator;		
  /**
   * \brief Constructor
   *
   * Creates a null header
   */
  DFCNHeader();
  ~DFCNHeader();
  /**
   * \param source the ip source to use in the underlying
   *        ip packet.
   * \param destination the ip destination to use in the
   *        underlying ip packet.
   * \param protocol the protocol number to use in the underlying
   *        ip packet.
   *
   * If you want to use udp checksums, you should call this
   * method prior to adding the header to a packet.
   */
	void  SetIdMessage (uint32_t level);
	/**
	 * \param port The source port for this MyMessageIDHeader
	 */
	uint32_t  GetIdMessage (void);
	
	void  SetListLength (uint32_t length);
	/**
	 * \param port The source port for this MyMessageIDHeader
	 */
	uint32_t  GetListLength (void) const;
	
	void  SetList (ListAddress list);
	/**
	 * \param port The source port for this MyMessageIDHeader
	 */ 
	ListAddress GetList (void);	
	
  void InitializeChecksum (Ipv4Address source, 
                           Ipv4Address destination,
                           uint8_t protocol);

  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);


private:

	
	uint32_t  m_IdMessage;
	uint32_t  m_ListLength;
	ListAddress m_neighborList;
	


};

} // namespace ns3

#endif /* DFCN_HEADER */
