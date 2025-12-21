import pandas as pd
import numpy as np
import ipaddress

def ip_to_numeric(ip_address):
    """
    Safely convert an IPv4 address string to its numeric representation.
    Returns None if the address is invalid.
    """
    try:
        # Handle cases where input might already be numeric or NaN
        if pd.isna(ip_address):
            return None
        return int(ipaddress.IPv4Address(ip_address))
    except (ValueError, TypeError):
        # Returns None for invalid formats (e.g., 'abc', '1.2.3')
        return None

def ip_range_to_country(ip_addresses, ip_df):    
    """Map a list of IP addresses to countries with error handling for malformed data."""
    
    # 1. Validate DataFrame Columns
    required_cols = ['lower_bound_ip_address', 'upper_bound_ip_address', 'country']
    if not all(col in ip_df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # 2. Prepare the IP ranges and drop rows with invalid range IPs
    ip_ranges = ip_df[required_cols].copy()
    ip_ranges['start_numeric'] = ip_ranges['lower_bound_ip_address'].apply(ip_to_numeric)
    ip_ranges['end_numeric'] = ip_ranges['upper_bound_ip_address'].apply(ip_to_numeric)
    
    # Remove ranges that couldn't be converted
    ip_ranges = ip_ranges.dropna(subset=['start_numeric', 'end_numeric'])

    # 3. Convert input IP addresses to numeric
    # We use a Series to keep track of indices even if some fail
    input_series = pd.Series(ip_addresses)
    ip_addresses_numeric = input_series.apply(ip_to_numeric)

    countries = []

    # 4. Map each IP address to its country
    for ip_num in ip_addresses_numeric:
        # Handle cases where the input IP itself was invalid
        if ip_num is None:
            countries.append('Invalid IP Format')
            continue

        # Find match in ranges
        match = ip_ranges[
            (ip_ranges['start_numeric'] <= ip_num) & 
            (ip_ranges['end_numeric'] >= ip_num)
        ]['country'].values

        if len(match) > 0:
            countries.append(match[0])
        else:
            countries.append('Unknown')

    return countries