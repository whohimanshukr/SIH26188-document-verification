'''Deterministic parser for standard two-line TD3 passport Machine Readable Zones (MRZ).'''

from typing import TypedDict


class ParsedMRZ(TypedDict):
    '''Normalized fields extracted from a TD3 MRZ.'''

    document_type: str
    issuing_country: str
    surname: str
    given_names: str
    passport_number: str
    nationality: str
    date_of_birth: str
    sex: str
    expiry_date: str


class MRZParser:
    '''Parse standard two-line TD3 (ICAO Doc 9303 Part 4) passport MRZ strings.

    TD3 structure consists of 2 lines of exactly 44 characters:
    Line 1:
      - Pos 0-1   (2):  Document code / type ('P<', 'P ', etc.)
      - Pos 2-4   (3):  Issuing country or organization (ISO 3166-1 alpha-3 code)
      - Pos 5-43 (39):  Name identifier: SURNAME<<GIVEN<NAMES (padded with '<')

    Line 2:
      - Pos 0-8   (9):  Document / passport number
      - Pos 9     (1):  Check digit for document number
      - Pos 10-12 (3):  Nationality (ISO 3166-1 alpha-3 code)
      - Pos 13-18 (6):  Date of birth (YYMMDD)
      - Pos 19    (1):  Check digit for date of birth
      - Pos 20    (1):  Sex ('M', 'F', or '<' for unspecified)
      - Pos 21-26 (6):  Date of expiry (YYMMDD)
      - Pos 27    (1):  Check digit for date of expiry
      - Pos 28-41 (14): Optional data / personal number
      - Pos 42    (1):  Check digit for optional data
      - Pos 43    (1):  Composite check digit
    '''

    TD3_LINE_LENGTH = 44

    def parse(self, lines: list[str]) -> ParsedMRZ:
        '''Parse two TD3 MRZ lines into structured fields.

        Args:
            lines: A list containing exactly two 44-character MRZ strings.

        Returns:
            ParsedMRZ: Dictionary containing normalized identity fields.

        Raises:
            ValueError: If line count or line length does not conform to TD3 specification.
        '''
        if not isinstance(lines, list) or len(lines) != 2:
            raise ValueError(
                f"TD3 MRZ requires exactly 2 lines, got {len(lines) if isinstance(lines, list) else type(lines).__name__}"
            )

        line1 = lines[0].strip()
        line2 = lines[1].strip()

        if len(line1) < 40 or len(line1) > self.TD3_LINE_LENGTH:
            raise ValueError(
                f"TD3 line 1 length must be between 40 and {self.TD3_LINE_LENGTH} characters, got {len(line1)}"
            )
        if len(line2) < 40 or len(line2) > self.TD3_LINE_LENGTH:
            raise ValueError(
                f"TD3 line 2 length must be between 40 and {self.TD3_LINE_LENGTH} characters, got {len(line2)}"
            )

        # In OCR/demo inputs, trailing '<' fillers are sometimes truncated or omit trailing pads.
        # Pad to standard TD3 length (44) with filler '<' characters.
        line1 = line1.ljust(self.TD3_LINE_LENGTH, "<")
        line2 = line2.ljust(self.TD3_LINE_LENGTH, "<")

        # Line 1 extraction
        # '<' is the standard ICAO filler character representing spaces or blank positions.
        document_type = line1[0:2].replace('<', '').strip()
        issuing_country = line1[2:5].replace('<', '').strip()

        # Name format: SURNAME<<GIVEN<NAMES
        # Primary identifier (surname) and secondary identifier (given names) are separated by '<<'.
        name_raw_part = line1[5:44]
        if '<<' in name_raw_part:
            surname_part, given_part = name_raw_part.split('<<', 1)
        else:
            surname_part, given_part = name_raw_part, ''

        # Single '<' represents a whitespace separator between compound names, trailing '<' are fillers.
        surname = surname_part.replace('<', ' ').strip()
        given_names = given_part.rstrip('<').replace('<', ' ').strip()

        # Line 2 extraction
        # Passport number occupies chars 0-8, padded with '<' on the right if shorter than 9 chars.
        passport_number = line2[0:9].rstrip('<')
        nationality = line2[10:13].replace('<', '').strip()

        # Dates are left in 6-digit YYMMDD format without guessing the century.
        date_of_birth = line2[13:19]
        sex_raw = line2[20]
        sex = 'Unspecified' if sex_raw == '<' else sex_raw
        expiry_date = line2[21:27]

        return {
            'document_type': document_type,
            'issuing_country': issuing_country,
            'surname': surname,
            'given_names': given_names,
            'passport_number': passport_number,
            'nationality': nationality,
            'date_of_birth': date_of_birth,
            'sex': sex,
            'expiry_date': expiry_date,
        }
