CURRENCY_SYMBOL = "₹"

def format_inr(amount):
    """Format amount as Indian Rupees with Indian number system (lakhs).
    
    Examples:
        1000 -> ₹1,000
        125000 -> ₹1,25,000
        1000000 -> ₹10,00,000
    """
    if amount is None:
        return f"{CURRENCY_SYMBOL}0"
    
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    
    str_amount = f"{amount:,.2f}"
    integer_part, _, decimal_part = str_amount.partition(".")
    
    integer_list = list(integer_part.replace(",", ""))
    
    result = []
    length = len(integer_list)
    
    first_part_length = length % 3
    if first_part_length == 0:
        first_part_length = 3
    
    for i, digit in enumerate(integer_list):
        if i > 0 and (i - first_part_length) % 2 == 0 and first_part_length != 3:
            result.append(",")
        elif i == first_part_length and first_part_length != 3:
            result.append(",")
        elif i > first_part_length and (i - first_part_length) % 2 == 0 and length > 4:
            result.append(",")
        result.append(digit)
    
    formatted_integer = "".join(result)
    formatted_amount = f"{CURRENCY_SYMBOL}{sign}{formatted_integer}.{decimal_part}"
    
    return formatted_amount


def format_inr_short(amount):
    """Format amount in short INR format (K, L, Cr).
    
    Examples:
        1000 -> ₹1K
        125000 -> ₹1.25L
        1000000 -> ₹10L
    """
    if amount is None:
        return f"{CURRENCY_SYMBOL}0"
    
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    
    if amount >= 10000000:
        return f"{CURRENCY_SYMBOL}{sign}{amount/10000000:.2f}Cr"
    elif amount >= 100000:
        return f"{CURRENCY_SYMBOL}{sign}{amount/100000:.2f}L"
    elif amount >= 1000:
        return f"{CURRENCY_SYMBOL}{sign}{amount/1000:.2f}K"
    else:
        return f"{CURRENCY_SYMBOL}{sign}{amount:.2f}"


def get_inr_symbol():
    """Return INR symbol."""
    return CURRENCY_SYMBOL