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

#ifndef DAGRS_HEADER_H
#define DAGRS_HEADER_H

#include <stdint.h>
#include <string>
#include "ns3/header.h"
#include "ns3/ipv4-address.h"
#include "ns3/wifi-mac-header.h"

namespace ns3 {
/**
 * \ingroup udp
 * \brief Packet header for UDP packets
 *
 * This class has fields corresponding to those in a network UDP header
 * (port numbers, payload size, checksum) as well as methods for serialization
 * to and deserialization from a byte buffer.
 */

class DAGRSHeader : public Header 
{
public:
	
//	enum MessageType {
//		IamToken         = 0,   Broadcasting the node is token
//		startMerge       = 1,   proposing starting merger 
//		ackStartingMerge = 2,   ack starting the merge 
//      MovingToken      = 3;   Traversing Token	
//	};
	
  /**
   * \brief Constructor
   *
   * Creates a null header
   */
  DAGRSHeader ();
  ~DAGRSHeader ();

  /**
   * \brief Enable checksum calculation for UDP 
   */
  void EnableChecksums (void);
  /**
   * \param port the destination port for this DAGRSHeader
   */
  void SetDestinationPort (uint16_t port);
  /**
   * \param port The source port for this DAGRSHeader
   */
  void SetSourcePort (uint16_t port);
  /**
   * \return The source port for this DAGRSHeader
   */
  uint16_t GetSourcePort (void) const;
  /**
   * \return the destination port for this DAGRSHeader
   */
  uint16_t GetDestinationPort (void) const;

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
  void InitializeChecksum (Ipv4Address source, 
                           Ipv4Address destination,
                           uint8_t protocol);

  static TypeId GetTypeId (void);
  virtual TypeId GetInstanceTypeId (void) const;
  virtual void Print (std::ostream &os) const;
  virtual uint32_t GetSerializedSize (void) const;
  virtual void Serialize (Buffer::Iterator start) const;
  virtual uint32_t Deserialize (Buffer::Iterator start);
  void SetMessageId (uint8_t id);
  uint8_t GetMessageId (void) const;	
  void SetSource (Ipv4Address id);
  Ipv4Address GetSource (void) const;
  /**
   * \brief Is the UDP checksum correct ?
   * \returns true if the checksum is correct, false otherwise.
   */
  bool IsChecksumOk (void) const;

private:
  uint16_t CalculateHeaderChecksum (uint16_t size) const;
  uint16_t m_sourcePort;
  uint16_t m_destinationPort;
  uint8_t m_messageId;
  Ipv4Address m_source;
  uint16_t m_payloadSize;
  
  Ipv4Address m_destination;
  uint8_t m_protocol;
  bool m_calcChecksum;
  bool m_goodChecksum;
	


};

} // namespace ns3

#endif /* UDP_HEADER */
