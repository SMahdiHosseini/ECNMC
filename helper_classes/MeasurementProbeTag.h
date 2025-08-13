//
// Created by Mahdi on 13.08.25.
//

#ifndef MEASUREMENT_PROBE_TAG_H
#define MEASUREMENT_PROBE_TAG_H

#include "ns3/tag.h"

using namespace ns3;
using namespace std;
/**
 * \brief Tag for the measurement probe
 */
class MeasurementProbeTag : public Tag
{
  public:
    MeasurementProbeTag();

    /**
     * \brief Set the tag's flag
     *
     * \param flag the flag
     */
    void SetFlag(bool flag);

    /**
     * \brief Get the tag's flag
     *
     * \returns the flag
     */
    bool GetFlag() const;

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    // inherited function, no need to doc.
    TypeId GetInstanceTypeId() const override;

    // inherited function, no need to doc.
    uint32_t GetSerializedSize() const override;

    // inherited function, no need to doc.
    void Serialize(TagBuffer i) const override;

    // inherited function, no need to doc.
    void Deserialize(TagBuffer i) override;

    // inherited function, no need to doc.
    void Print(std::ostream& os) const override;

  private:
    bool m_isProbe; //!< the flag to indicate if this is a measurement probe
}; 

#endif // MEASUREMENT_PROBE_TAG_H
