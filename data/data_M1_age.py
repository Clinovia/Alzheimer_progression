import pandas as pd
import os

def parse_ptdob(dob_str):
    """
    Parse PTDOB using two rules:
    1. Month-Number (e.g., Apr-31) → use month, day = 15
    2. Number-Month (e.g., 28-Jan) → use day and month

    Returns: {'day': int, 'month': int} or None
    """
    if pd.isna(dob_str) or str(dob_str).strip() == '':
        return None

    dob_str = str(dob_str).strip()
    if '-' not in dob_str:
        return None

    parts = dob_str.split('-')
    if len(parts) != 2:
        return None

    part1, part2 = parts[0].strip(), parts[1].strip()

    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    # Rule 1: Month-Number → e.g., Apr-31, Feb-33
    if part1.capitalize() in month_map and part2.isdigit():
        month = month_map[part1.capitalize()]
        day = 15  # default day
        return {'day': day, 'month': month}

    # Rule 2: Number-Month → e.g., 28-Jan, 3-Mar
    if part1.isdigit() and part2.capitalize() in month_map:
        day = int(part1)
        month = month_map[part2.capitalize()]
        # Validate day
        if 1 <= day <= 31:
            return {'day': day, 'month': month}

    return None  # No match

def parse_visdate(date_str):
    """Parse VISDATE like '10/18/17' or '11/27/2017'"""
    if pd.isna(date_str) or str(date_str).strip() == '':
        return None
    for fmt in ('%m/%d/%y', '%m/%d/%Y'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(date_str, errors='coerce')  # fallback

def calculate_age_at_visit(df):
    """
    Add AGE column using:
    - PTDOB (parsed with your two rules)
    - PTDOBYY (birth year)
    - VISDATE (visit date)
    """
    print("🧮 Calculating AGE at visit using your two rules...")

    # Parse PTDOB
    parsed = df['PTDOB'].apply(parse_ptdob)
    df['DOB_day'] = parsed.apply(lambda x: x['day'] if x else None)
    df['DOB_month'] = parsed.apply(lambda x: x['month'] if x else None)

    # Parse VISDATE
    df['VISDATE_DT'] = df['VISDATE'].astype(str).apply(parse_visdate)

    # Create full birth date using PTDOBYY, DOB_month, DOB_day
    birth_data = pd.DataFrame({
        'year': df['PTDOBYY'],
        'month': df['DOB_month'],
        'day': df['DOB_day']
    })

    # Convert to datetime (errors='coerce' handles invalid dates)
    df['BIRTH_DT'] = pd.to_datetime(birth_data, errors='coerce')

    # ✅ CORRECT WAY TO COMPUTE AGE IN YEARS
    # Use .dt.days after subtracting two datetime columns
    time_diff = df['VISDATE_DT'] - df['BIRTH_DT']
    df['AGE'] = (time_diff.dt.days / 365.25).round().astype(int)

    # Clean up
    df = df.drop(columns=['DOB_day', 'DOB_month', 'VISDATE_DT', 'BIRTH_DT'])

    # Report
    valid_age_count = df['AGE'].notna().sum()
    print(f"✅ Successfully computed AGE for {valid_age_count} subjects")
    if 'AGE' in df.columns and df['AGE'].notna().any():
        min_age = df['AGE'].min()
        max_age = df['AGE'].max()
        print(f"📊 Age range: {min_age} to {max_age} years")
    else:
        print("❌ No valid AGE computed")

    return df

# =============================================
# ✅ Run the script
# =============================================

if __name__ == "__main__":
    INPUT_FILE = 'data_M1.csv'
    OUTPUT_FILE = 'data_M1_with_age.csv'

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    print(f"📄 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    required_cols = ['PTDOB', 'PTDOBYY', 'VISDATE']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Add AGE
    df = calculate_age_at_visit(df)

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n🎉 Saved final dataset with AGE to: {OUTPUT_FILE}")