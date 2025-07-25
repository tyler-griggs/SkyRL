"""
uv run --extra dev --isolated pytest tests/cpu/generators/test_utils.py
"""

import pytest
from skyrl_train.generators.utils import apply_overlong_filtering


@pytest.mark.parametrize(
    "loss_masks,stop_reasons,expected_masks",
    [
        # Test case 1: No length stop reasons - masks should remain unchanged
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
            ["stop", "stop", "stop"],
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
        ),
        # Test case 2: All length stop reasons - all masks should be zeroed
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
            ["length", "length", "length"],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0]],
        ),
        # Test case 3: Mixed stop reasons - only length masks should be zeroed
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0, 1]],
            ["stop", "length", "stop"],
            [[1, 1, 0, 1], [0, 0, 0, 0], [1, 0, 1, 0, 1]],
        ),
        # Test case 4: Different stop reasons including length
        (
            [[1, 1], [1, 0, 1], [0, 1, 1, 1]],
            ["eos", "length", "abort"],
            [[1, 1], [0, 0, 0], [0, 1, 1, 1]],
        ),
        # Test case 5: Empty lists
        ([], [], []),
    ],
)
def test_apply_overlong_filtering(loss_masks, stop_reasons, expected_masks):
    """
    Test the apply_overlong_filtering function which implements DAPO Overlong Filtering.
    
    This function should zero-out every token's mask whenever the stop_reason was 'length'
    (i.e. truncated), while leaving other masks unchanged.
    """
    result = apply_overlong_filtering(loss_masks, stop_reasons)
    
    assert result == expected_masks, f"Expected {expected_masks}, but got {result}"
    
    # Verify that the original inputs are not modified (immutability check)
    assert len(result) == len(loss_masks), "Result should have same length as input"
    
    # Check that each individual mask is processed correctly
    for i, (original_mask, stop_reason, expected_mask) in enumerate(zip(loss_masks, stop_reasons, expected_masks)):
        if stop_reason == "length":
            # Should be all zeros with same length as original
            assert result[i] == [0] * len(original_mask), f"Mask {i} should be all zeros for length stop reason"
        else:
            # Should be unchanged
            assert result[i] == original_mask, f"Mask {i} should be unchanged for non-length stop reason"


def test_apply_overlong_filtering_immutability():
    """
    Test that apply_overlong_filtering doesn't modify the original input lists.
    """
    original_loss_masks = [[1, 1, 0, 1], [0, 1, 1]]
    original_stop_reasons = ["stop", "length"]
    
    # Create copies to compare against later
    loss_masks_copy = [mask[:] for mask in original_loss_masks]  # Deep copy of lists
    stop_reasons_copy = original_stop_reasons[:]  # Shallow copy is fine for strings
    
    result = apply_overlong_filtering(original_loss_masks, original_stop_reasons)
    
    # Verify original inputs are unchanged
    assert original_loss_masks == loss_masks_copy, "Original loss_masks should not be modified"
    assert original_stop_reasons == stop_reasons_copy, "Original stop_reasons should not be modified"
    
    # Verify result is correct
    expected = [[1, 1, 0, 1], [0, 0, 0]]  # Second mask zeroed due to "length"
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "loss_masks,stop_reasons",
    [
        # Test case 1: More loss_masks than stop_reasons
        ([[1, 1], [0, 1]], ["stop"]),
        # Test case 2: More stop_reasons than loss_masks  
        ([[1, 1]], ["stop", "length"]),
        # Test case 3: Empty loss_masks but non-empty stop_reasons
        ([], ["stop"]),
        # Test case 4: Non-empty loss_masks but empty stop_reasons
        ([[1, 0]], []),
    ],
)
def test_apply_overlong_filtering_length_mismatch_assertion(loss_masks, stop_reasons):
    """
    Test that apply_overlong_filtering raises AssertionError when loss_masks and stop_reasons 
    have different lengths.
    """
    with pytest.raises(AssertionError, match="loss_masks and stop_reasons must have the same length"):
        apply_overlong_filtering(loss_masks, stop_reasons) 