import unittest
from macrosynergy.panel.panel_calculator import _get_xcats_used


TEST_CASES = {
    1: {
        "calc_str": "NEW1 = XR + CRY",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    2: {
        "calc_str": "NEW2 = XR - CRY",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    3: {
        "calc_str": "SINGLE1 = XR - iUSD_INFL",
        "output": {
            "all_xcats_used": ["INFL", "XR"],
            "singles_used": ["iUSD_INFL"],
            "single_cids": ["USD"],
        },
    },
    4: {
        "calc_str": "NEW3 = XR * CRY",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    5: {
        "calc_str": "NEW4 = XR / CRY",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    6: {
        "calc_str": "NEW5 = (XR+CRY)/2",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    7: {
        "calc_str": "NEW6 = ( XR - CRY ) * 100",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    8: {
        "calc_str": "NEW7 = XR ** 2",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    9: {
        "calc_str": "NEW8 = np.abs( XR )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    10: {
        "calc_str": "NEW9 = np.sign( XR )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    11: {
        "calc_str": "NEW10 = np.sqrt( np.abs( XR ) )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    12: {
        "calc_str": "NEW11 = np.log( np.abs( XR ) + 1 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    13: {
        "calc_str": "NEW12 = np.exp( XR / 10 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    14: {
        "calc_str": "NEW13 = np.tanh( XR )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    15: {
        "calc_str": "NEW14 = np.sin( XR )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    16: {
        "calc_str": "NEW15 = np.cos( XR )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    17: {
        "calc_str": "NEW16 = np.arctan( XR )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    18: {
        "calc_str": "NEW17 = np.maximum( XR , 0 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    19: {
        "calc_str": "NEW18 = np.minimum( XR , 0 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    20: {
        "calc_str": "NEW19 = np.maximum( XR , CRY )",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    21: {
        "calc_str": "NEW20 = np.minimum( XR , CRY )",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    22: {
        "calc_str": "NEW21 = ( XR + 0.5 ) * CRY",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    23: {
        "calc_str": "NEW22 = ( XR - 0.5 ) / ( np.abs( CRY ) + 1 )",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    24: {
        "calc_str": "NEW23 = ( XR * CRY ) / ( np.abs( XR ) + np.abs( CRY ) + 1 )",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    25: {
        "calc_str": "NEW24 = ( XR - CRY ) / ( np.abs( XR - CRY ) + 1 )",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    26: {
        "calc_str": "NEW25 = ( GROWTH - INFL )",
        "output": {
            "all_xcats_used": ["GROWTH", "INFL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    27: {
        "calc_str": "NEW26=(GROWTH-INFL)/(np.abs(INFL)+1)",
        "output": {
            "all_xcats_used": ["GROWTH", "INFL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    28: {
        "calc_str": "NEW27 = ( CARRY + VALUE + MOM ) / 3",
        "output": {
            "all_xcats_used": ["CARRY", "MOM", "VALUE"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    29: {
        "calc_str": "NEW28 = ( CARRY * 0.5 ) + ( VALUE * 0.3 ) + ( MOM * 0.2 )",
        "output": {
            "all_xcats_used": ["CARRY", "MOM", "VALUE"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    30: {
        "calc_str": "NEW29=(CARRY-VALUE)/(np.abs(CARRY)+np.abs(VALUE)+1)",
        "output": {
            "all_xcats_used": ["CARRY", "VALUE"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    31: {
        "calc_str": "NEW30 = ( MOM - VOL )",
        "output": {
            "all_xcats_used": ["MOM", "VOL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    32: {
        "calc_str": "NEW31 = MOM / ( VOL + 1 )",
        "output": {
            "all_xcats_used": ["MOM", "VOL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    33: {
        "calc_str": "NEW32 = np.abs( MOM ) / ( np.abs( VOL ) + 1 )",
        "output": {
            "all_xcats_used": ["MOM", "VOL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    34: {
        "calc_str": "NEW33 = ( RET > 0 ) * 1",
        "output": {
            "all_xcats_used": ["RET"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    35: {
        "calc_str": "NEW34 = ( RET < 0 ) * 1",
        "output": {
            "all_xcats_used": ["RET"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    36: {
        "calc_str": "NEW35 = ( RET >= 0 ) * 2 - 1",
        "output": {
            "all_xcats_used": ["RET"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    37: {
        "calc_str": "NEW36 = ( XR > CRY ) * 1",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    38: {
        "calc_str": "NEW37 = ( XR > 0 ) & ( CRY > 0 )",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    39: {
        "calc_str": "NEW38 = ( XR > 0 ) | ( CRY > 0 )",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    40: {
        "calc_str": "NEW39 = ~( XR > 0 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    41: {
        "calc_str": "NEW40 = ( np.abs( XR ) > 2 ) * 1",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    42: {
        "calc_str": "SINGLE2 = GROWTH - iEUR_GROWTH",
        "output": {
            "all_xcats_used": ["GROWTH"],
            "singles_used": ["iEUR_GROWTH"],
            "single_cids": ["EUR"],
        },
    },
    43: {
        "calc_str": "SINGLE3 = INFL - iJPY_INFL",
        "output": {
            "all_xcats_used": ["INFL"],
            "singles_used": ["iJPY_INFL"],
            "single_cids": ["JPY"],
        },
    },
    44: {
        "calc_str": "SINGLE4 = np.sqrt( np.abs( iUSD_XR ) )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": ["iUSD_XR"],
            "single_cids": ["USD"],
        },
    },
    45: {
        "calc_str": "SINGLE5 = np.log( np.abs( iEUR_INFL ) + 1 )",
        "output": {
            "all_xcats_used": ["INFL"],
            "singles_used": ["iEUR_INFL"],
            "single_cids": ["EUR"],
        },
    },
    46: {
        "calc_str": "SINGLE6 = ( XR + iUSD_XR ) / 2",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": ["iUSD_XR"],
            "single_cids": ["USD"],
        },
    },
    47: {
        "calc_str": "SINGLE7=(CRY-iUSD_CRY)/(np.abs(iUSD_CRY)+1)",
        "output": {
            "all_xcats_used": ["CRY"],
            "singles_used": ["iUSD_CRY"],
            "single_cids": ["USD"],
        },
    },
    48: {
        "calc_str": "SINGLE8 = ( XR - iUSD_XR ) / ( np.abs( XR ) + 1 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": ["iUSD_XR"],
            "single_cids": ["USD"],
        },
    },
    49: {
        "calc_str": "SINGLE9 = ( XR - iUSD_XR ) / ( np.abs( iUSD_XR ) + 1 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": ["iUSD_XR"],
            "single_cids": ["USD"],
        },
    },
    50: {
        "calc_str": "SINGLE10 = ( XR - iUSD_XR ) / ( np.abs( XR - iUSD_XR ) + 1 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": ["iUSD_XR"],
            "single_cids": ["USD"],
        },
    },
    51: {
        "calc_str": "CPIH_SJA_P3M3ML3ARX = CPIH_SJA_P3M3ML3AR - INFTARGET_NSA",
        "output": {
            "all_xcats_used": ["CPIH_SJA_P3M3ML3AR", "INFTARGET_NSA"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    52: {
        "calc_str": "CPIC_SJA_P6M6ML6ARX = CPIC_SJA_P6M6ML6AR - INFTARGET_NSA",
        "output": {
            "all_xcats_used": ["CPIC_SJA_P6M6ML6AR", "INFTARGET_NSA"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    53: {
        "calc_str": "FXBLACK = ( FXTARGETED_NSA + FXUNTRADABLE_NSA ) > 0",
        "output": {
            "all_xcats_used": ["FXTARGETED_NSA", "FXUNTRADABLE_NSA"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    54: {
        "calc_str": "TS1 = XR.shift( 1 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    55: {
        "calc_str": "TS2 = XR.shift( 1 ) - XR.shift( 5 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    56: {
        "calc_str": "TS3 = XR.diff( 1 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    57: {
        "calc_str": "TS4 = XR.pct_change( 5 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    58: {
        "calc_str": "TS5 = ( XR.shift( 1 ) + CRY.shift( 1 ) ) / 2",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    59: {
        "calc_str": "TS6 = ( XR.shift( 1 ) - CRY.shift( 1 ) ) / ( np.abs( CRY.shift( 1 ) ) + 1 )",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    60: {
        "calc_str": "TS7 = XR.rolling( 20 ).mean()",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    61: {
        "calc_str": "TS8 = XR.rolling( 20 ).std()",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    62: {
        "calc_str": "TS9 = ( XR.rolling( 20 ).mean() ) / ( np.abs( XR.rolling( 252 ).mean() ) + 1 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    63: {
        "calc_str": "TS10 = ( XR - XR.rolling( 252 ).mean() ) / ( XR.rolling( 252 ).std() + 1e-9 )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    64: {
        "calc_str": "TS11 = XR.ewm( span=20 ).mean()",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    65: {
        "calc_str": "TS12 = ( XR.ewm( span=20 ).mean() - XR.ewm( span=60 ).mean() )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    66: {
        "calc_str": "TS13 = SIGNAL.rank( axis=1 , pct=True )",
        "output": {
            "all_xcats_used": ["SIGNAL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    67: {
        "calc_str": "TS14 = ( SIGNAL - SIGNAL.mean( axis=1 ) ) / ( SIGNAL.std( axis=1 ) + 1e-9 )",
        "output": {
            "all_xcats_used": ["SIGNAL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    68: {
        "calc_str": "TS15 = SIGNAL.clip( lower=-2 , upper=2 )",
        "output": {
            "all_xcats_used": ["SIGNAL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    69: {
        "calc_str": "TS16 = SIGNAL.fillna( 0 )",
        "output": {
            "all_xcats_used": ["SIGNAL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    70: {
        "calc_str": "TS17 = SIGNAL.where( np.abs( SIGNAL ) < 3 )",
        "output": {
            "all_xcats_used": ["SIGNAL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    71: {
        "calc_str": "TS18 = ( SIGNAL.where( SIGNAL > 0 ) ).fillna( 0 )",
        "output": {
            "all_xcats_used": ["SIGNAL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    72: {
        "calc_str": "TS19 = ( XR.shift( 1 ) > XR.shift( 252 ) ) * 1",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    73: {
        "calc_str": "TS20 = ( XR.shift( 1 ) > 0 ) & ( CRY.shift( 1 ) > 0 )",
        "output": {
            "all_xcats_used": ["CRY", "XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    74: {
        "calc_str": "SINGLE1 = XR - iUSD_XR",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": ["iUSD_XR"],
            "single_cids": ["USD"],
        },
    },
    75: {
        "calc_str": "TS21 = XR.ewm(span=20).mean()",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    76: {
        "calc_str": "TS22 = XR.pct_change(periods=5, fill_method='pad')",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    77: {
        "calc_str": "TS23 = XR.rolling(window=20, min_periods=5).mean()",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    78: {
        "calc_str": "TS24 = XR.rolling(window=20, min_periods=5).std(ddof=0)",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    79: {
        "calc_str": "TS25 = XR.rank(axis=0, pct=1)",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    80: {
        "calc_str": "TS26 = SIGNAL.where(np.abs(SIGNAL) < 3, other=np.nan)",
        "output": {
            "all_xcats_used": ["SIGNAL"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    81: {
        "calc_str": "TS27 = XR.where(XR > 0, other=0)",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    82: {
        "calc_str": "TS28 = XR.mask(XR < 0, other=0)",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    83: {
        "calc_str": "XR_LO = np.clip( XR , a_min=0, a_max=np.inf )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    84: {
        "calc_str": "XR_CLEAN = np.where( np.isinf( XR ), 0, XR )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    85: {
        "calc_str": "XR_CLEAN = np.where( np.isnan( XR ), CRY, XR )",
        "output": {
            "all_xcats_used": ["XR", "CRY"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    86: {
        "calc_str": "XR_CLIP = np.clip( XR - iUSD_INFL , a_min=0, a_max=np.inf )",
        "output": {
            "all_xcats_used": ["XR", "INFL"],
            "singles_used": ["iUSD_INFL"],
            "single_cids": ["USD"],
        },
    },
    87: {
        "calc_str": "GROWTH_LO = np.clip( GROWTH , a_min=0, a_max=np.inf )",
        "output": {
            "all_xcats_used": ["GROWTH"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    88: {
        "calc_str": "XR_BOUND = np.clip( XR , a_min=-np.inf, a_max=np.inf )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    89: {
        "calc_str": "XR_CLEAN = np.where( np.isinf( XR ) | np.isnan( XR ), 0, XR )",
        "output": {
            "all_xcats_used": ["XR"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    90: {
        "calc_str": "XR_VALID = np.where( np.isfinite( XR ), XR, CRY )",
        "output": {
            "all_xcats_used": ["XR", "CRY"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    91: {
        "calc_str": "XR_IMG = np.imag( XR ) + np.real( CRY )",
        "output": {
            "all_xcats_used": ["XR", "CRY"],
            "singles_used": [],
            "single_cids": [],
        },
    },
    92: {
        "calc_str": "NEW1 = np.clip( GROWTH - iEUR_INFL , a_min=-np.inf, a_max=np.inf )",
        "output": {
            "all_xcats_used": ["GROWTH", "INFL"],
            "singles_used": ["iEUR_INFL"],
            "single_cids": ["EUR"],
        },
    },
}


class TestPanelCalculatorCalcStrings(unittest.TestCase):
    def test_get_xcats_used(self):
        test_cases = TEST_CASES.copy()
        for key in test_cases:
            calc_str: str = test_cases[key]["calc_str"].replace(" ", "")
            expected_output = test_cases[key]["output"]
            lhs, rhs = calc_str.split("=", 1)
            ops = {lhs.strip(): rhs.strip()}

            (
                all_xcats_used,
                singles_used,
                single_cids,
            ) = _get_xcats_used(ops)

            self.assertEqual(
                set(all_xcats_used), set(expected_output["all_xcats_used"])
            )
            self.assertEqual(set(singles_used), set(expected_output["singles_used"]))
            self.assertEqual(set(single_cids), set(expected_output["single_cids"]))

        print(f"Tested {len(test_cases)} calculation strings successfully.")


if __name__ == "__main__":
    unittest.main()
