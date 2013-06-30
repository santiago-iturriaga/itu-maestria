/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2005,2006 INRIA
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
#ifndef MAC_HIGH_MYBEACONING_H
#define MAC_HIGH_MYBEACONING_H

#include <stdint.h>
#include "ns3/random-variable.h"
#include <map>
#include "ns3/mac48-address.h"
#include "ns3/callback.h"
#include "ns3/packet.h"
#include "ns3/nstime.h"
#include "supported-rates.h"
#include "wifi-remote-station-manager.h"
#include "wifi-mac.h"
#include "ns3/EnergyEfficient-application.h"

namespace ns3 {

class WifiMacHeader;
class DcaTxop;
class WifiPhy;
class WifiNetDevice;	
class DcfManager;
class MacRxMiddle;
class MacLow;
//class Application;
	//class RandomVariable;

/**
 * \brief non-QoS AP state machine
 *
 * Handle association, dis-association and authentication,
 * of STAs within an IBSS.
 * This class uses two output queues, each of which is server by
 * a single DCF
 *   - the highest priority DCF serves the queue which contains
 *     only beacons.
 *   - the lowest priority DCF serves the queue which contains all
 *     other frames, including user data frames.
 */
class MyBeaconingWifiMac : public WifiMac
{
public:
  static TypeId GetTypeId (void);

  MyBeaconingWifiMac ();
  ~MyBeaconingWifiMac ();

  // inherited from WifiMac.
  virtual void SetSlot (Time slotTime);
  virtual void SetSifs (Time sifs);
  virtual void SetEifsNoDifs (Time eifsNoDifs);
  virtual void SetAckTimeout (Time ackTimeout);
  virtual void SetCtsTimeout (Time ctsTimeout);
  virtual void SetPifs (Time pifs);
  virtual Time GetSlot (void) const;
  virtual Time GetSifs (void) const;
  virtual Time GetEifsNoDifs (void) const;
  virtual Time GetAckTimeout (void) const;
  virtual Time GetCtsTimeout (void) const;
  virtual Time GetPifs (void) const;
  virtual void SetWifiPhy (Ptr<WifiPhy> phy);
  virtual void SetWifiRemoteStationManager (Ptr<WifiRemoteStationManager> stationManager);
  virtual void Enqueue (Ptr<const Packet> packet, Mac48Address to, Mac48Address from);
  virtual void Enqueue (Ptr<const Packet> packet, Mac48Address to);
  virtual bool SupportsSendFrom (void) const;
  virtual void SetForwardUpCallback (Callback<void,Ptr<Packet>, Mac48Address, Mac48Address> upCallback);
//Added by Patricia Ruiz for call back when detecting neighbors	
//  virtual void MySetForwardUpCallback (Callback<void,Ptr<Packet>, Mac48Address,double> neighborCallback);
//  virtual void MySetForwardUpNeighborLostCallback (Callback<void,Ptr<Packet>, Mac48Address> neighborLostCallback);

  virtual void SetLinkUpCallback (Callback<void> linkUp);
  virtual void SetLinkDownCallback (Callback<void> linkDown);
  virtual Mac48Address GetAddress (void) const;
  virtual Ssid GetSsid (void) const;
  virtual void SetAddress (Mac48Address address);
  virtual void SetSsid (Ssid ssid);
  virtual Mac48Address GetBssid (void) const;

  /**
   * \param interval the interval between two beacon transmissions.
   */
  void SetBeaconInterval (Time interval);
  /**
   * \returns the interval between two beacon transmissions.
   */
  Time GetBeaconInterval (void) const;
  /**
   * Start beacon transmission immediately.
   */
  void StartBeaconing (void);

 // void SetForwardNeighborListCallback (Callback<void,Ptr<Packet>, Mac48Address>neighborCallback);	
 // void SetForwardNeighborLostListCallback (Callback<void,Ptr<Packet>, Mac48Address>neighborLostCallback);	
	

private:
  void Receive (Ptr<Packet> packet, WifiMacHeader const *hdr);
  void ForwardUp (Ptr<Packet> packet, Mac48Address from, Mac48Address to);
  void ForwardDown (Ptr<const Packet> packet, Mac48Address from, Mac48Address to);
  void TxOk (WifiMacHeader const &hdr);
  void TxFailed (WifiMacHeader const &hdr);
  void SendProbeResp (Mac48Address to);
  void SendAssocResp (Mac48Address to, bool success);
  void SendOneBeacon (void);
  SupportedRates GetSupportedRates (void) const;
  void SetBeaconGeneration (bool enable);
  bool GetBeaconGeneration (void) const;
  virtual void DoDispose (void);
  MyBeaconingWifiMac (const MyBeaconingWifiMac & ctor_arg);
  MyBeaconingWifiMac &operator = (const MyBeaconingWifiMac &o);
  Ptr<DcaTxop> GetDcaTxop (void) const;
  void SetDcaTxop (Ptr<DcaTxop> dcaTxop);
  void ProcessBeacon( Ptr<Packet> packet, Mac48Address addrFrom, Mac48Address addrTo);
  void DecreaseBeaconCount();
  Ptr<DcaTxop> m_dca;
  Ptr<DcaTxop> m_beaconDca;
  Ptr<WifiRemoteStationManager> m_stationManager;
  Ptr<WifiPhy> m_phy;
  Callback<void, Ptr<Packet>,Mac48Address, Mac48Address> m_upCallback;

  Time m_beaconInterval;

  DcfManager *m_dcfManager;
  MacRxMiddle *m_rxMiddle;
  Ptr<MacLow> m_low;
  Ssid m_ssid;
  EventId m_beaconEvent;
	
	// My own vbles (Patricia Ruiz)
  typedef	std::map <Mac48Address,uint16_t> MacAddrMap;
  typedef   std::map<Mac48Address, uint16_t>::iterator MacAddrMapIterator;	
  uint16_t      m_numberBeaconLost;
  MacAddrMap    m_neighborList;
  Callback<void,Ptr<Packet>, Mac48Address, double> m_neighborCallback; 
  Callback<void,Ptr<Packet>, Mac48Address> m_neighborLostCallback;
	
//  void ForwardNeighborList(Ptr<Packet> packet, Mac48Address from, double rxPwDbm);
//  void ForwardNeighborLostList(Ptr<Packet> packet, Mac48Address from);
	
  double m_rxPwDbm;
  TracedCallback<Ptr<const Packet>, Mac48Address > m_neighborLostTraceSource;
  TracedCallback<Ptr<const Packet>, Mac48Address,double  > m_newNeighborTraceSource;


};

} // namespace ns3


#endif /* MAC_HIGH_myBeaconing_H */
