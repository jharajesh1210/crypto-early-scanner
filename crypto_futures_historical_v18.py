# ============================================================
# crypto_futures_historical_v18.py
#
# COINDCX FUTURES HISTORICAL V18
# ROLLING WALK-FORWARD EXPECTANCY ENGINE
#
# Historical research only.
# NO Telegram.
# NO real orders.
# ============================================================

import numpy as np
import pandas as pd

import crypto_futures_historical_v17 as v17


# ============================================================
# SETTINGS
# ============================================================

PAIR = "B-BTC_USDT"

ROUND_TRIP_COST_PCT = 0.10

# Rolling walk-forward:
# 120 days train
# 30 days validation
# move forward 30 days each fold

TRAIN_DAYS = 120
TEST_DAYS = 30
STEP_DAYS = 30

# Minimum samples

MIN_TRAIN_TRADES = 30
MIN_TEST_TRADES = 8

# ATR TP/SL grid
# Kept deliberately small to reduce overfitting

TP_GRID = [
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
]

SL_GRID = [
    0.75,
    1.00,
    1.25,
    1.50,
]

# Number of best training configurations
# allowed into each fold's test period

TOP_TRAIN_CONFIGS = 3

# Final reliability requirements

PASS_MIN_FOLDS = 5
PASS_MIN_POSITIVE_FOLDS = 4
PASS_MIN_TOTAL_TEST_TRADES = 50
PASS_MIN_NET_AVG = 0.0
PASS_MIN_PF = 1.15
PASS_MAX_DD = 8.0

WATCH_MIN_FOLDS = 4
WATCH_MIN_POSITIVE_FOLDS = 3
WATCH_MIN_TOTAL_TEST_TRADES = 30
WATCH_MIN_NET_AVG = 0.0
WATCH_MIN_PF = 1.00


# ============================================================
# OUTPUT FILES
# ============================================================

EVENT_FILE = "crypto_futures_v18_events.csv"

FOLD_FILE = "crypto_futures_v18_folds.csv"

CONFIG_FILE = "crypto_futures_v18_configs.csv"

SUMMARY_FILE = "crypto_futures_v18_summary.csv"


# ============================================================
# PERFORMANCE
# ============================================================

def performance(trades):

    if trades is None or trades.empty:
        return None

    returns = pd.to_numeric(
        trades["NET_RETURN_%"],
        errors="coerce",
    ).dropna()

    if returns.empty:
        return None

    wins = int(
        (trades["OUTCOME"] == "WIN").sum()
    )

    losses = int(
        (trades["OUTCOME"] == "LOSS").sum()
    )

    time_exits = int(
        (trades["OUTCOME"] == "TIME_EXIT").sum()
    )

    decisive = wins + losses

    if decisive > 0:

        win_rate = (
            wins
            /
            decisive
            *
            100.0
        )

    else:

        win_rate = np.nan

    positive = returns[
        returns > 0
    ]

    negative = returns[
        returns < 0
    ]

    gross_profit = float(
        positive.sum()
    )

    gross_loss = abs(
        float(
            negative.sum()
        )
    )

    if gross_loss > 0:

        pf = (
            gross_profit
            /
            gross_loss
        )

    elif gross_profit > 0:

        pf = 999.0

    else:

        pf = 0.0

    equity = returns.cumsum()

    peak = equity.cummax()

    drawdown = (
        equity
        -
        peak
    )

    max_dd = abs(
        float(
            drawdown.min()
        )
    )

    return {

        "TRADES":
            len(trades),

        "DECISIVE":
            decisive,

        "WINS":
            wins,

        "LOSSES":
            losses,

        "TIME_EXITS":
            time_exits,

        "WIN_RATE_%":
            round(
                win_rate,
                2,
            )
            if pd.notna(win_rate)
            else np.nan,

        "NET_AVG_RETURN_%":
            round(
                float(
                    returns.mean()
                ),
                4,
            ),

        "TOTAL_NET_RETURN_%":
            round(
                float(
                    returns.sum()
                ),
                4,
            ),

        "PROFIT_FACTOR":
            round(
                pf,
                4,
            ),

        "MAX_DRAWDOWN_%":
            round(
                max_dd,
                4,
            ),
    }


# ============================================================
# V18 CANDIDATES
# ============================================================

def build_candidates():

    rows = []

    for side in [
        "LONG",
        "SHORT",
    ]:

        definitions = [

            (
                "BASE",
                None,
                None,
            ),

            (
                "LOW_VOL",
                "LOW_VOL",
                None,
            ),

            (
                "NORMAL_VOL",
                "NORMAL_VOL",
                None,
            ),

            (
                "HIGH_VOL",
                "HIGH_VOL",
                None,
            ),

            (
                "MEDIUM_TREND",
                None,
                "MEDIUM",
            ),

            (
                "STRONG_TREND",
                None,
                "STRONG",
            ),

            (
                "NORMAL_MEDIUM",
                "NORMAL_VOL",
                "MEDIUM",
            ),

            (
                "NORMAL_STRONG",
                "NORMAL_VOL",
                "STRONG",
            ),

            (
                "HIGH_MEDIUM",
                "HIGH_VOL",
                "MEDIUM",
            ),

            (
                "HIGH_STRONG",
                "HIGH_VOL",
                "STRONG",
            ),
        ]

        for (
            name,
            volatility,
            strength,
        ) in definitions:

            rows.append(
                {
                    "NAME":
                        f"V18_{side}_{name}",

                    "SIDE":
                        side,

                    "VOL":
                        volatility,

                    "STRENGTH":
                        strength,
                }
            )

    return rows


# ============================================================
# FILTER EVENTS
# ============================================================

def filter_events(
    events,
    candidate,
):

    data = events[
        events["SIDE"]
        ==
        candidate["SIDE"]
    ].copy()

    if candidate["VOL"] is not None:

        data = data[
            data["VOL_REGIME"]
            ==
            candidate["VOL"]
        ].copy()

    if candidate["STRENGTH"] is not None:

        data = data[
            data["TREND_STRENGTH"]
            ==
            candidate["STRENGTH"]
        ].copy()

    return data


# ============================================================
# RUN CONFIGURATION
# ============================================================

def run_config(
    market,
    events,
    candidate,
    tp_atr,
    sl_atr,
):

    selected = filter_events(
        events,
        candidate,
    )

    if selected.empty:
        return None, pd.DataFrame()

    trades = v17.resimulate(
        market,
        selected,
        candidate["SIDE"],
        tp_atr,
        sl_atr,
    )

    stats = performance(
        trades
    )

    return stats, trades


# ============================================================
# CREATE CALENDAR WALK-FORWARD FOLDS
# ============================================================

def make_walk_forward_folds(events):

    if events.empty:
        return []

    start = (
        events["TIME"]
        .min()
        .normalize()
    )

    end = (
        events["TIME"]
        .max()
        .normalize()
    )

    folds = []

    fold_number = 1

    train_start = start

    while True:

        train_end = (
            train_start
            +
            pd.Timedelta(
                days=TRAIN_DAYS
            )
        )

        test_start = train_end

        test_end = (
            test_start
            +
            pd.Timedelta(
                days=TEST_DAYS
            )
        )

        if test_end > end:
            break

        train_mask = (
            (events["TIME"] >= train_start)
            &
            (events["TIME"] < train_end)
        )

        test_mask = (
            (events["TIME"] >= test_start)
            &
            (events["TIME"] < test_end)
        )

        train = (
            events[
                train_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        test = (
            events[
                test_mask
            ]
            .copy()
            .reset_index(drop=True)
        )

        if (
            not train.empty
            and
            not test.empty
        ):

            folds.append(
                {
                    "FOLD":
                        fold_number,

                    "TRAIN_START":
                        train_start,

                    "TRAIN_END":
                        train_end,

                    "TEST_START":
                        test_start,

                    "TEST_END":
                        test_end,

                    "TRAIN":
                        train,

                    "TEST":
                        test,
                }
            )

            fold_number += 1

        train_start = (
            train_start
            +
            pd.Timedelta(
                days=STEP_DAYS
            )
        )

    return folds


# ============================================================
# TRAINING SEARCH
# ============================================================

def search_training(
    market,
    train_events,
    candidates,
):

    rows = []

    for candidate in candidates:

        candidate_events = filter_events(
            train_events,
            candidate,
        )

        if (
            len(candidate_events)
            <
            MIN_TRAIN_TRADES
        ):

            continue

        for tp_atr in TP_GRID:

            for sl_atr in SL_GRID:

                stats, trades = run_config(
                    market,
                    train_events,
                    candidate,
                    tp_atr,
                    sl_atr,
                )

                if stats is None:
                    continue

                if (
                    stats["TRADES"]
                    <
                    MIN_TRAIN_TRADES
                ):

                    continue

                rows.append(
                    {
                        "CANDIDATE":
                            candidate["NAME"],

                        "SIDE":
                            candidate["SIDE"],

                        "VOL_REGIME":
                            candidate["VOL"],

                        "TREND_STRENGTH":
                            candidate["STRENGTH"],

                        "TP_ATR":
                            tp_atr,

                        "SL_ATR":
                            sl_atr,

                        **stats,
                    }
                )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    # Main priority:
    # positive expectancy,
    # then PF,
    # then sample size.
    #
    # Win rate is NOT the primary sorting variable.

    result = (
        result
        .sort_values(
            [
                "NET_AVG_RETURN_%",
                "PROFIT_FACTOR",
                "TRADES",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# FIND CANDIDATE
# ============================================================

def find_candidate(
    candidates,
    name,
):

    for candidate in candidates:

        if candidate["NAME"] == name:
            return candidate

    return None


# ============================================================
# RUN WALK FORWARD
# ============================================================

def run_walk_forward(
    market,
    events,
    candidates,
):

    folds = make_walk_forward_folds(
        events
    )

    print()
    print("=" * 120)

    print(
        "V18 ROLLING WALK-FORWARD"
    )

    print("=" * 120)

    print(
        "TOTAL FOLDS:",
        len(folds)
    )

    fold_rows = []

    config_rows = []

    for fold in folds:

        fold_number = fold["FOLD"]

        train = fold["TRAIN"]

        test = fold["TEST"]

        print()
        print("-" * 120)

        print(
            "FOLD:",
            fold_number
        )

        print(
            "TRAIN:",
            fold["TRAIN_START"],
            "->",
            fold["TRAIN_END"],
        )

        print(
            "TEST:",
            fold["TEST_START"],
            "->",
            fold["TEST_END"],
        )

        print(
            "TRAIN EVENTS:",
            len(train)
        )

        print(
            "TEST EVENTS:",
            len(test)
        )

        training_results = search_training(
            market,
            train,
            candidates,
        )

        if training_results.empty:

            print(
                "NO TRAINING CONFIGURATION."
            )

            continue

        # Keep top configs separately
        # for LONG and SHORT.

        shortlist_parts = []

        for side in [
            "LONG",
            "SHORT",
        ]:

            side_rows = (
                training_results[
                    training_results[
                        "SIDE"
                    ]
                    ==
                    side
                ]
                .head(
                    TOP_TRAIN_CONFIGS
                )
                .copy()
            )

            if not side_rows.empty:

                shortlist_parts.append(
                    side_rows
                )

        if not shortlist_parts:
            continue

        shortlist = pd.concat(
            shortlist_parts,
            ignore_index=True,
        )

        for _, train_config in (
            shortlist.iterrows()
        ):

            candidate = find_candidate(
                candidates,
                train_config[
                    "CANDIDATE"
                ],
            )

            if candidate is None:
                continue

            test_events = filter_events(
                test,
                candidate,
            )

            if (
                len(test_events)
                <
                MIN_TEST_TRADES
            ):

                continue

            test_stats, test_trades = (
                run_config(
                    market,
                    test,
                    candidate,
                    float(
                        train_config[
                            "TP_ATR"
                        ]
                    ),
                    float(
                        train_config[
                            "SL_ATR"
                        ]
                    ),
                )
            )

            if test_stats is None:
                continue

            if (
                test_stats["TRADES"]
                <
                MIN_TEST_TRADES
            ):

                continue

            config_id = (
                f'{candidate["NAME"]}'
                f'_TP{float(train_config["TP_ATR"]):.2f}'
                f'_SL{float(train_config["SL_ATR"]):.2f}'
            )

            row = {

                "FOLD":
                    fold_number,

                "CONFIG_ID":
                    config_id,

                "CANDIDATE":
                    candidate["NAME"],

                "SIDE":
                    candidate["SIDE"],

                "VOL_REGIME":
                    candidate["VOL"],

                "TREND_STRENGTH":
                    candidate["STRENGTH"],

                "TP_ATR":
                    float(
                        train_config[
                            "TP_ATR"
                        ]
                    ),

                "SL_ATR":
                    float(
                        train_config[
                            "SL_ATR"
                        ]
                    ),

                "TRAIN_START":
                    fold["TRAIN_START"],

                "TRAIN_END":
                    fold["TRAIN_END"],

                "TEST_START":
                    fold["TEST_START"],

                "TEST_END":
                    fold["TEST_END"],

                "TRAIN_TRADES":
                    train_config[
                        "TRADES"
                    ],

                "TRAIN_WIN_RATE_%":
                    train_config[
                        "WIN_RATE_%"
                    ],

                "TRAIN_NET_AVG_%":
                    train_config[
                        "NET_AVG_RETURN_%"
                    ],

                "TRAIN_PF":
                    train_config[
                        "PROFIT_FACTOR"
                    ],

                "TEST_TRADES":
                    test_stats[
                        "TRADES"
                    ],

                "TEST_DECISIVE":
                    test_stats[
                        "DECISIVE"
                    ],

                "TEST_WINS":
                    test_stats[
                        "WINS"
                    ],

                "TEST_LOSSES":
                    test_stats[
                        "LOSSES"
                    ],

                "TEST_TIME_EXITS":
                    test_stats[
                        "TIME_EXITS"
                    ],

                "TEST_WIN_RATE_%":
                    test_stats[
                        "WIN_RATE_%"
                    ],

                "TEST_NET_AVG_%":
                    test_stats[
                        "NET_AVG_RETURN_%"
                    ],

                "TEST_TOTAL_NET_%":
                    test_stats[
                        "TOTAL_NET_RETURN_%"
                    ],

                "TEST_PF":
                    test_stats[
                        "PROFIT_FACTOR"
                    ],

                "TEST_MAX_DD_%":
                    test_stats[
                        "MAX_DRAWDOWN_%"
                    ],
            }

            fold_rows.append(
                row
            )

            config_rows.append(
                row.copy()
            )

        print()

        current = pd.DataFrame(
            [
                x
                for x in fold_rows
                if x["FOLD"]
                ==
                fold_number
            ]
        )

        if not current.empty:

            display_columns = [
                "CANDIDATE",
                "SIDE",
                "TP_ATR",
                "SL_ATR",
                "TEST_TRADES",
                "TEST_WIN_RATE_%",
                "TEST_NET_AVG_%",
                "TEST_PF",
            ]

            print(
                current[
                    display_columns
                ]
                .sort_values(
                    "TEST_NET_AVG_%",
                    ascending=False,
                )
                .to_string(
                    index=False
                )
            )

    return (
        pd.DataFrame(
            fold_rows
        ),
        pd.DataFrame(
            config_rows
        ),
    )


# ============================================================
# AGGREGATE OUT-OF-SAMPLE CONFIGURATIONS
# ============================================================

def aggregate_configs(
    config_results
):

    if config_results.empty:
        return pd.DataFrame()

    rows = []

    for (
        config_id,
        group,
    ) in config_results.groupby(
        "CONFIG_ID"
    ):

        group = (
            group
            .sort_values("FOLD")
            .reset_index(drop=True)
        )

        folds = len(group)

        total_trades = int(
            group[
                "TEST_TRADES"
            ].sum()
        )

        positive_folds = int(
            (
                group[
                    "TEST_NET_AVG_%"
                ]
                >
                0
            ).sum()
        )

        weighted_net = (
            (
                group[
                    "TEST_NET_AVG_%"
                ]
                *
                group[
                    "TEST_TRADES"
                ]
            ).sum()
            /
            max(
                total_trades,
                1,
            )
        )

        total_wins = int(
            group[
                "TEST_WINS"
            ].sum()
        )

        total_losses = int(
            group[
                "TEST_LOSSES"
            ].sum()
        )

        decisive = (
            total_wins
            +
            total_losses
        )

        if decisive > 0:

            win_rate = (
                total_wins
                /
                decisive
                *
                100.0
            )

        else:

            win_rate = np.nan

        # Approximate combined PF from fold totals.
        # We also require consistency across folds,
        # so one good fold cannot dominate.

        pf_values = pd.to_numeric(
            group["TEST_PF"],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        median_pf = (
            float(
                pf_values.median()
            )
            if not pf_values.empty
            else 0.0
        )

        mean_pf = (
            float(
                pf_values.mean()
            )
            if not pf_values.empty
            else 0.0
        )

        worst_net = float(
            group[
                "TEST_NET_AVG_%"
            ].min()
        )

        median_net = float(
            group[
                "TEST_NET_AVG_%"
            ].median()
        )

        max_dd = float(
            group[
                "TEST_MAX_DD_%"
            ].max()
        )

        positive_ratio = (
            positive_folds
            /
            folds
            if folds > 0
            else 0.0
        )

        first = group.iloc[0]

        status = "REJECT"

        if (
            folds
            >=
            PASS_MIN_FOLDS

            and
            positive_folds
            >=
            PASS_MIN_POSITIVE_FOLDS

            and
            total_trades
            >=
            PASS_MIN_TOTAL_TEST_TRADES

            and
            weighted_net
            >
            PASS_MIN_NET_AVG

            and
            median_pf
            >=
            PASS_MIN_PF

            and
            max_dd
            <=
            PASS_MAX_DD
        ):

            status = "PASS"

        elif (
            folds
            >=
            WATCH_MIN_FOLDS

            and
            positive_folds
            >=
            WATCH_MIN_POSITIVE_FOLDS

            and
            total_trades
            >=
            WATCH_MIN_TOTAL_TEST_TRADES

            and
            weighted_net
            >
            WATCH_MIN_NET_AVG

            and
            median_pf
            >=
            WATCH_MIN_PF
        ):

            status = "WATCH"

        rows.append(
            {
                "STATUS":
                    status,

                "CONFIG_ID":
                    config_id,

                "CANDIDATE":
                    first[
                        "CANDIDATE"
                    ],

                "SIDE":
                    first[
                        "SIDE"
                    ],

                "VOL_REGIME":
                    first[
                        "VOL_REGIME"
                    ],

                "TREND_STRENGTH":
                    first[
                        "TREND_STRENGTH"
                    ],

                "TP_ATR":
                    first[
                        "TP_ATR"
                    ],

                "SL_ATR":
                    first[
                        "SL_ATR"
                    ],

                "FOLDS_TESTED":
                    folds,

                "POSITIVE_FOLDS":
                    positive_folds,

                "POSITIVE_FOLD_RATIO":
                    round(
                        positive_ratio,
                        4,
                    ),

                "TOTAL_TEST_TRADES":
                    total_trades,

                "TOTAL_WINS":
                    total_wins,

                "TOTAL_LOSSES":
                    total_losses,

                "OOS_WIN_RATE_%":
                    round(
                        win_rate,
                        2,
                    )
                    if pd.notna(
                        win_rate
                    )
                    else np.nan,

                "OOS_WEIGHTED_NET_AVG_%":
                    round(
                        weighted_net,
                        4,
                    ),

                "OOS_MEDIAN_NET_AVG_%":
                    round(
                        median_net,
                        4,
                    ),

                "OOS_WORST_FOLD_NET_%":
                    round(
                        worst_net,
                        4,
                    ),

                "OOS_MEDIAN_PF":
                    round(
                        median_pf,
                        4,
                    ),

                "OOS_MEAN_PF":
                    round(
                        mean_pf,
                        4,
                    ),

                "MAX_FOLD_DRAWDOWN_%":
                    round(
                        max_dd,
                        4,
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    status_order = {
        "PASS": 0,
        "WATCH": 1,
        "REJECT": 2,
    }

    result[
        "_STATUS_ORDER"
    ] = result[
        "STATUS"
    ].map(
        status_order
    )

    result = (
        result
        .sort_values(
            [
                "_STATUS_ORDER",
                "OOS_WEIGHTED_NET_AVG_%",
                "OOS_MEDIAN_PF",
                "TOTAL_TEST_TRADES",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )
        .drop(
            columns=[
                "_STATUS_ORDER"
            ]
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 120)

    print(
        "COINDCX FUTURES HISTORICAL V18"
    )

    print(
        "ROLLING WALK-FORWARD EXPECTANCY ENGINE"
    )

    print("=" * 120)

    print()

    print(
        "PAIR:",
        PAIR
    )

    print(
        "ENTRY TIMEFRAME: 5 MINUTES"
    )

    print(
        "TREND TIMEFRAME: 1 HOUR"
    )

    print()

    print(
        "TRAIN WINDOW:",
        TRAIN_DAYS,
        "DAYS"
    )

    print(
        "TEST WINDOW:",
        TEST_DAYS,
        "DAYS"
    )

    print(
        "ROLL STEP:",
        STEP_DAYS,
        "DAYS"
    )

    print()

    print(
        "ROUND-TRIP COST:",
        ROUND_TRIP_COST_PCT,
        "%"
    )

    print()

    print(
        "PRIMARY GOAL:"
    )

    print(
        "OUT-OF-SAMPLE POSITIVE EXPECTANCY"
    )

    print(
        "+ PROFIT FACTOR"
    )

    print(
        "+ REPEATABILITY ACROSS TIME"
    )

    print()

    print(
        "WIN RATE IS SECONDARY."
    )

    print()

    print(
        "NO REAL ORDERS WILL BE PLACED."
    )

    # --------------------------------------------------------
    # Download using proven V17 downloader
    # --------------------------------------------------------

    (
        df_5m,
        df_1h,
        actual_days,
    ) = v17.download_history()

    print()
    print(
        "ACTUAL HISTORY:",
        actual_days,
        "DAYS"
    )

    # --------------------------------------------------------
    # Prepare market using V17 indicators
    # --------------------------------------------------------

    market = v17.prepare_market(
        df_5m,
        df_1h,
    )

    print()
    print(
        "MARKET DATA READY."
    )

    print(
        "USABLE 5M CANDLES:",
        len(market)
    )

    # --------------------------------------------------------
    # Collect base events using V17 signal logic
    # --------------------------------------------------------

    print()
    print(
        "Collecting V18 base events..."
    )

    events = v17.collect_events(
        market
    )

    if events is None or events.empty:

        raise RuntimeError(
            "No V18 base events found."
        )

    events["TIME"] = pd.to_datetime(
        events["TIME"],
        utc=True,
        errors="coerce",
    )

    events = (
        events
        .dropna(
            subset=[
                "TIME"
            ]
        )
        .sort_values(
            "TIME"
        )
        .reset_index(drop=True)
    )

    print()
    print(
        "TOTAL EVENTS:",
        len(events)
    )

    print(
        "LONG:",
        int(
            (
                events["SIDE"]
                ==
                "LONG"
            ).sum()
        )
    )

    print(
        "SHORT:",
        int(
            (
                events["SIDE"]
                ==
                "SHORT"
            ).sum()
        )
    )

    print()

    print(
        "EVENT RANGE:"
    )

    print(
        events["TIME"].min(),
        "->",
        events["TIME"].max(),
    )

    # --------------------------------------------------------
    # Candidates
    # --------------------------------------------------------

    candidates = build_candidates()

    print()
    print(
        "REGIME CANDIDATES:",
        len(candidates)
    )

    # --------------------------------------------------------
    # Rolling walk-forward
    # --------------------------------------------------------

    (
        fold_results,
        config_results,
    ) = run_walk_forward(
        market,
        events,
        candidates,
    )

    # --------------------------------------------------------
    # Aggregate true OOS results
    # --------------------------------------------------------

    summary = aggregate_configs(
        config_results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    events.to_csv(
        EVENT_FILE,
        index=False,
    )

    fold_results.to_csv(
        FOLD_FILE,
        index=False,
    )

    config_results.to_csv(
        CONFIG_FILE,
        index=False,
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 120)

    print(
        "V18 FILES SAVED"
    )

    print("=" * 120)

    print(
        EVENT_FILE
    )

    print(
        FOLD_FILE
    )

    print(
        CONFIG_FILE
    )

    print(
        SUMMARY_FILE
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print("=" * 160)

    print(
        "V18 OUT-OF-SAMPLE WALK-FORWARD SUMMARY"
    )

    print("=" * 160)

    if summary.empty:

        print()
        print(
            "NO CONFIGURATION HAD"
        )

        print(
            "ENOUGH OUT-OF-SAMPLE DATA."
        )

        print()
        print(
            "V18 RESULT = NO STRATEGY"
        )

    else:

        print()

        print(
            summary
            .head(30)
            .to_string(
                index=False
            )
        )

        pass_count = int(
            (
                summary["STATUS"]
                ==
                "PASS"
            ).sum()
        )

        watch_count = int(
            (
                summary["STATUS"]
                ==
                "WATCH"
            ).sum()
        )

        reject_count = int(
            (
                summary["STATUS"]
                ==
                "REJECT"
            ).sum()
        )

        print()
        print("=" * 120)

        print(
            "PASS:",
            pass_count
        )

        print(
            "WATCH:",
            watch_count
        )

        print(
            "REJECT:",
            reject_count
        )

        print("=" * 120)

        if pass_count > 0:

            print()
            print(
                "V18 PASS CANDIDATES:"
            )

            print()

            print(
                summary[
                    summary[
                        "STATUS"
                    ]
                    ==
                    "PASS"
                ]
                .to_string(
                    index=False
                )
            )

            print()
            print(
                "IMPORTANT:"
            )

            print(
                "PASS DOES NOT MEAN"
            )

            print(
                "REAL-MONEY READY."
            )

            print()

            print(
                "NEXT STEP:"
            )

            print(
                "FORWARD PAPER TRADING."
            )

        elif watch_count > 0:

            print()
            print(
                "NO V18 PASS."
            )

            print()

            print(
                "WATCH candidates exist,"
            )

            print(
                "but they are not ready"
            )

            print(
                "for live trading."
            )

        else:

            print()
            print(
                "NO V18 PASS OR WATCH."
            )

            print()

            print(
                "NO STRATEGY SHOULD"
            )

            print(
                "BE ACTIVATED."
            )

    print()
    print("=" * 120)

    print(
        "V18 INTERPRETATION"
    )

    print("=" * 120)

    print()

    print(
        "PASS:"
    )

    print(
        "Repeated positive out-of-sample"
    )

    print(
        "performance across rolling periods."
    )

    print()

    print(
        "WATCH:"
    )

    print(
        "Some positive evidence,"
    )

    print(
        "but not strong enough."
    )

    print()

    print(
        "REJECT:"
    )

    print(
        "Do not use for live trading."
    )

    print()

    print(
        "V18 DOES NOT FORCE"
    )

    print(
        "70%-80% WIN RATE."
    )

    print()

    print(
        "EXPECTANCY + PROFIT FACTOR"
    )

    print(
        "COME FIRST."
    )

    print()

    print(
        "NO REAL ORDERS WERE PLACED."
    )

    print()
    print("=" * 120)

    print(
        "FUTURES HISTORICAL V18 COMPLETE"
    )

    print("=" * 120)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
