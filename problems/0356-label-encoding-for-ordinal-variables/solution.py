def label_encode_ordinal(values: list, order: list) -> list:
    """
    Encode ordinal categorical values to integers based on specified order.
    """
    # Edge case: If the input list is empty, return an empty list
    if not values:
        return []
        
    # 1. Create a dictionary mapping each category to its rank/index
    # Example: ['small', 'medium', 'large'] -> {'small': 0, 'medium': 1, 'large': 2}
    order_map = {category: index for index, category in enumerate(order)}
    
    # 2. Iterate through the values and encode them
    encoded_values = []
    for val in values:
        # The .get() method looks up the value in the dictionary.
        # If the value is not found, it returns the default fallback (-1).
        encoded_index = order_map.get(val, -1)
        encoded_values.append(encoded_index)
        
    return encoded_values