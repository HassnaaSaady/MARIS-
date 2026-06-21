def vessel_label(code) -> str:
    if code is None:
        return "Unknown Vessel Type"

    try:
        int_code = int(float(code))
    except (ValueError, TypeError):
        return str(code) if code else "Unknown Vessel Type"

    if 70 <= int_code <= 79:
        return "Cargo"

    if 80 <= int_code <= 89:
        return "Tanker"

    if 60 <= int_code <= 69:
        return "Passenger"

    if int_code == 30:
        return "Fishing"

    if int_code in (31, 32, 52):
        return "Towing / Tug"

    if int_code == 35:
        return "Military"

    if int_code == 36:
        return "Sailing"

    if int_code == 37:
        return "Pleasure Craft"

    return "Unknown Vessel Type"