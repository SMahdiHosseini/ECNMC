//
// Created by Mahdi on 13.08.25.
//

#include "MeasurementProbeTag.h"


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
