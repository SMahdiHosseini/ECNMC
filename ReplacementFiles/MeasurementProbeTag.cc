//
// Created by Mahdi on 13.08.25.
//

#include "MeasurementProbeTag.h"

namespace ns3
{

MeasurementProbeTag::MeasurementProbeTag()
{
}

void
MeasurementProbeTag::SetFlag(bool flag)
{
    m_isProbe = flag;
}

bool
MeasurementProbeTag::GetFlag() const
{
    return m_isProbe;
}

TypeId
MeasurementProbeTag::GetTypeId()
{
    static TypeId tid = TypeId("ns3::MeasurementProbeTag")
                            .SetParent<Tag>()
                            .SetGroupName("Network")
                            .AddConstructor<MeasurementProbeTag>();
    return tid;
}

TypeId
MeasurementProbeTag::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
MeasurementProbeTag::GetSerializedSize() const
{
    return sizeof(bool);
}

void
MeasurementProbeTag::Serialize(TagBuffer i) const
{
    i.WriteU8(m_isProbe);
}

void
MeasurementProbeTag::Deserialize(TagBuffer i)
{
    m_isProbe = i.ReadU8();
}

void
MeasurementProbeTag::Print(std::ostream& os) const
{
    os << "MEASUREMENT_PROBE = " << m_isProbe;
}   

MeasurementProbeTagWithBits::MeasurementProbeTagWithBits()
{
}

void
MeasurementProbeTagWithBits::SetBitFlag(uint32_t flag)
{
    m_bits.push_back(flag);
}

std::vector<uint32_t>
MeasurementProbeTagWithBits::GetBitsFlag() const
{
    return m_bits;
}

TypeId
MeasurementProbeTagWithBits::GetTypeId()
{
    static TypeId tid = TypeId("ns3::MeasurementProbeTagWithBits")
                            .SetParent<MeasurementProbeTag>()
                            .SetGroupName("Network")
                            .AddConstructor<MeasurementProbeTagWithBits>();
    return tid;
}

TypeId
MeasurementProbeTagWithBits::GetInstanceTypeId() const
{
    return GetTypeId();
}

uint32_t
MeasurementProbeTagWithBits::GetSerializedSize() const
{
    return MeasurementProbeTag::GetSerializedSize() + ((m_bits.size() + 1) * sizeof(uint32_t));
}

void
MeasurementProbeTagWithBits::Serialize(TagBuffer i) const
{
    MeasurementProbeTag::Serialize(i);
    i.WriteU32(m_bits.size());
    for (const auto& bit : m_bits)
    {
        i.WriteU32(bit);
    }
}

void
MeasurementProbeTagWithBits::Deserialize(TagBuffer i)
{
    MeasurementProbeTag::Deserialize(i);
    uint32_t size = i.ReadU32();
    m_bits.clear();
    for (uint32_t j = 0; j < size; ++j)
    {
        m_bits.push_back(i.ReadU32());
    }
}

void
MeasurementProbeTagWithBits::Print(std::ostream& os) const
{
    MeasurementProbeTag::Print(os);
    os << "MEASUREMENT_PROBE_BITS = ";
    for (const auto& bit : m_bits)
    {
        os << bit << " ";
    }
}

}