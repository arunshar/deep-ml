def birthday_problem(n: int, days: int = 365) -> float:
    """
    Calculate the probability that at least two people share the same birthday.
    """
    # 1. Edge Cases
    # If there is 1 or fewer people, a shared birthday is impossible.
    if n <= 1:
        return 0.0
        
    # By the Pigeonhole Principle, if there are more people than days, 
    # a shared birthday is 100% guaranteed.
    if n > days:
        return 1.0
        
    # 2. Calculate the probability of NO matches
    prob_no_match = 1.0
    
    for i in range(n):
        # For each new person, the number of "available" unshared birthdays 
        # decreases by 1 (days - i)
        prob_no_match *= (days - i) / days
        
    # 3. Calculate the probability of AT LEAST ONE match
    prob_match = 1.0 - prob_no_match
    
    # 4. Return the result rounded to 4 decimal places
    return round(prob_match, 4)