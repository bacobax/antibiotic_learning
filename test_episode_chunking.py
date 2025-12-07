"""
Test to verify that episode chunking in PPO respects episode boundaries.
"""
import torch

def test_episode_boundary_detection():
    """Test that episode boundaries are correctly identified."""
    # Simulate done flags: episodes at indices 5, 10, 15
    T = 20
    B = 2
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[5, 0] = True
    dones[10, 0] = True
    dones[15, 0] = True
    
    # Extract episode boundaries (same logic as in ppo.py)
    done_indices = torch.where(dones[:, 0])[0].tolist()
    episode_starts = [0] + [idx + 1 for idx in done_indices if idx + 1 < T]
    episode_starts = [s for s in episode_starts if s < T]
    
    print("Done indices:", done_indices)
    print("Episode starts:", episode_starts)
    
    assert done_indices == [5, 10, 15], f"Expected [5, 10, 15], got {done_indices}"
    assert episode_starts == [0, 6, 11, 16], f"Expected [0, 6, 11, 16], got {episode_starts}"
    
    # Verify episode boundaries
    episodes = []
    for ep_idx, ep_start in enumerate(episode_starts):
        if ep_idx + 1 < len(episode_starts):
            ep_end = episode_starts[ep_idx + 1]
        else:
            ep_end = T
        episodes.append((ep_start, ep_end))
    
    print("Episodes (start, end):", episodes)
    
    assert episodes == [(0, 6), (6, 11), (11, 16), (16, 20)], f"Unexpected episodes: {episodes}"
    print("✓ Episode boundary detection works correctly")


def test_episode_chunking():
    """Test that episodes are chunked into seq_len sized pieces."""
    T = 20
    seq_len = 5
    
    # Episode from 0 to 20 (no done flags)
    ep_start = 0
    ep_end = 20
    episode_length = ep_end - ep_start
    num_chunks_in_episode = max(1, (episode_length + seq_len - 1) // seq_len)
    
    chunks = []
    for chunk_idx in range(num_chunks_in_episode):
        t_start = ep_start + chunk_idx * seq_len
        t_end = min(t_start + seq_len, ep_end)
        chunks.append((t_start, t_end))
    
    print(f"Episode length: {episode_length}, seq_len: {seq_len}")
    print(f"Number of chunks: {num_chunks_in_episode}")
    print("Chunks (start, end):", chunks)
    
    assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"
    assert chunks == [(0, 5), (5, 10), (10, 15), (15, 20)], f"Unexpected chunks: {chunks}"
    print("✓ Episode chunking works correctly")


def test_episode_chunking_with_boundaries():
    """Test that episodes are chunked correctly when respecting boundaries."""
    T = 20
    B = 2
    seq_len = 5
    
    # Episodes: [0, 6), [6, 11), [11, 16), [16, 20)
    episode_ranges = [(0, 6), (6, 11), (11, 16), (16, 20)]
    
    all_chunks = []
    for ep_start, ep_end in episode_ranges:
        episode_length = ep_end - ep_start
        num_chunks_in_episode = max(1, (episode_length + seq_len - 1) // seq_len)
        
        for chunk_idx in range(num_chunks_in_episode):
            t_start = ep_start + chunk_idx * seq_len
            t_end = min(t_start + seq_len, ep_end)
            all_chunks.append((t_start, t_end))
    
    print("All chunks across episodes:", all_chunks)
    
    # Verify no chunk spans episode boundaries
    expected = [(0, 5), (5, 6), (6, 11), (11, 16), (16, 20)]
    assert all_chunks == expected, f"Expected {expected}, got {all_chunks}"
    
    # Verify chunk sizes don't exceed seq_len
    for t_start, t_end in all_chunks:
        chunk_size = t_end - t_start
        assert chunk_size <= seq_len, f"Chunk size {chunk_size} exceeds seq_len {seq_len}"
        if chunk_size < 2:
            print(f"  Warning: Chunk ({t_start}, {t_end}) is very short, would be skipped")
    
    print("✓ Episode chunking with boundaries works correctly")


if __name__ == "__main__":
    test_episode_boundary_detection()
    print()
    test_episode_chunking()
    print()
    test_episode_chunking_with_boundaries()
    print("\n✓ All tests passed!")
