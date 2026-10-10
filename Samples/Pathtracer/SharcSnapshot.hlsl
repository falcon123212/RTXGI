/*
 * Compact SHaRC snapshots: save the valid cache entries of a lighting state and restore
 * them in place when the same state comes back.
 *
 * A record is 24 bytes: hash table index, packed sample data, hash key, resolved radiance.
 * Only entries with a key and at least one sample are kept, so a snapshot is a few MB
 * instead of a full copy of the hash, accumulation and resolved buffers.
 */

struct SharcSnapshotRecord
{
    uint index;
    uint sampleData;
    uint2 key;
    uint2 radiance;     // 4 x fp16: radiance rgb + sample count (SharcPackedData.radianceData)
};

RWStructuredBuffer<uint2>               u_SharcHashEntries  : register(u0, space3);
RWStructuredBuffer<uint4>               u_SharcResolved     : register(u1, space3);
RWStructuredBuffer<SharcSnapshotRecord> u_SnapshotRecords   : register(u2, space3);
RWStructuredBuffer<uint>                u_SnapshotCounter   : register(u3, space3);

uint GetEntryCount()
{
    uint count, stride;
    u_SharcHashEntries.GetDimensions(count, stride);
    return count;
}

uint GetRecordCapacity()
{
    uint count, stride;
    u_SnapshotRecords.GetDimensions(count, stride);
    return count;
}

bool IsValidEntry(uint index, out uint2 key, out uint4 resolved)
{
    key = u_SharcHashEntries[index];
    resolved = u_SharcResolved[index];
    return (key.x | key.y) != 0 && (resolved.y >> 16) != 0;   // key present and sample count > 0
}

[numthreads(256, 1, 1)]
void sharcSnapshotCount(uint index : SV_DispatchThreadID)
{
    if (index >= GetEntryCount())
        return;

    uint2 key;
    uint4 resolved;
    if (IsValidEntry(index, key, resolved))
        InterlockedAdd(u_SnapshotCounter[0], 1);
}

[numthreads(256, 1, 1)]
void sharcSnapshotPack(uint index : SV_DispatchThreadID)
{
    if (index >= GetEntryCount())
        return;

    uint2 key;
    uint4 resolved;
    if (!IsValidEntry(index, key, resolved))
        return;

    uint slot;
    InterlockedAdd(u_SnapshotCounter[0], 1, slot);
    if (slot >= GetRecordCapacity())
        return;

    SharcSnapshotRecord record;
    record.index = index;
    record.sampleData = resolved.z;
    record.key = key;
    record.radiance = resolved.xy;
    u_SnapshotRecords[slot] = record;
}

// The caller clears the hash, accumulation and resolved buffers before this pass.
[numthreads(256, 1, 1)]
void sharcSnapshotRestore(uint index : SV_DispatchThreadID)
{
    if (index >= GetRecordCapacity())
        return;

    SharcSnapshotRecord record = u_SnapshotRecords[index];
    u_SharcHashEntries[record.index] = record.key;
    // Keep the accumulated frame count, reset the stale frame count
    u_SharcResolved[record.index] = uint4(record.radiance, record.sampleData & 0xFFFFu, 0);
}
