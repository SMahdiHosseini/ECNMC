//
// Created by Mahdi on 13.08.25.
// copied in src/network/utils/MeasurementProbeTag.h and add to the cmake file
//

#ifndef MEASUREMENT_PROBE_TAG_H
#define MEASUREMENT_PROBE_TAG_H

#include "ns3/tag.h"

namespace ns3
{

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

  protected:
    bool m_isProbe; //!< the flag to indicate if this is a measurement probe
};

/**
 * \brief Tag for the measurement probe
 */
class MeasurementProbeTagWithBits : public MeasurementProbeTag
{
  public:
    MeasurementProbeTagWithBits();

    /**
     * \brief Set the tag's flag
     *
     * \param flag the flag
     */
    void SetBitFlag(uint32_t flag);

    /**
     * \brief Get the tag's flag
     *
     * \returns the flag
     */
    std::vector<uint32_t> GetBitsFlag() const;

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

    /**
     *  Assignment operator
     * \param [in] o tag to assign.
     * \return The tag.
     */
    inline MeasurementProbeTagWithBits& operator=(const MeasurementProbeTagWithBits& o)
    {
        m_isProbe = o.m_isProbe;
        m_bits = o.m_bits; // Copy the bits vector
        return *this;
    }

  private:
    std::vector<uint32_t> m_bits; //!< the flags to indicate the measurement probe bits
};
} // namespace ns3
#endif // MEASUREMENT_PROBE_TAG_H
