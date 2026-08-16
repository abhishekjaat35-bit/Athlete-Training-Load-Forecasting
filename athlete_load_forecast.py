import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print("=" * 80)
print("              ATHLETE TRAINING LOAD FORECASTING SYSTEM")
print("=" * 80)


# ------------------------------------------
# Load Data
# ------------------------------------------

data = pd.read_csv(
    "athlete_load_forecast_data.csv"
)

data["Date"] = pd.to_datetime(
    data["Date"]
)

data = data.sort_values(
    ["Athlete", "Date"]
).reset_index(drop=True)


# ------------------------------------------
# Data Validation
# ------------------------------------------

print("\n" + "=" * 80)
print("DATA VALIDATION")
print("=" * 80)

print(f"Rows           : {len(data)}")
print(f"Columns        : {len(data.columns)}")
print(
    f"Missing values : "
    f"{data.isnull().sum().sum()}"
)
print(
    f"Athletes       : "
    f"{data['Athlete'].nunique()}"
)


# ------------------------------------------
# Rolling Training Load
# ------------------------------------------

data["Rolling_Load_3"] = (
    data.groupby("Athlete")["Training_Load"]
    .transform(
        lambda x: x.rolling(
            window=3,
            min_periods=1
        ).mean()
    )
)

data["Rolling_Load_5"] = (
    data.groupby("Athlete")["Training_Load"]
    .transform(
        lambda x: x.rolling(
            window=5,
            min_periods=1
        ).mean()
    )
)


# ------------------------------------------
# Load Change
# ------------------------------------------

data["Previous_Load"] = (
    data.groupby("Athlete")["Training_Load"]
    .shift(1)
)

data["Load_Change_%"] = (
    (
        data["Training_Load"]
        -
        data["Previous_Load"]
    )
    /
    data["Previous_Load"]
) * 100

data["Load_Change_%"] = (
    data["Load_Change_%"]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ------------------------------------------
# Athlete Baseline
# ------------------------------------------

data["Athlete_Baseline"] = (
    data.groupby("Athlete")["Training_Load"]
    .transform("mean")
)


# ------------------------------------------
# Forecast Function
# ------------------------------------------

def forecast_next_load(athlete_data):

    athlete_data = athlete_data.sort_values(
        "Date"
    ).reset_index(drop=True)

    y = athlete_data["Training_Load"].values

    x = np.arange(len(y))

    if len(y) < 2:
        return float(y[-1])

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    next_x = len(y)

    prediction = (
        slope * next_x
        +
        intercept
    )

    return max(
        0,
        prediction
    )


# ------------------------------------------
# Generate Forecasts
# ------------------------------------------

forecast_rows = []

for athlete in data["Athlete"].unique():

    athlete_data = data[
        data["Athlete"] == athlete
    ].copy()

    predicted_load = (
        forecast_next_load(
            athlete_data
        )
    )

    latest_load = (
        athlete_data
        .sort_values("Date")
        .iloc[-1]["Training_Load"]
    )

    baseline = (
        athlete_data["Training_Load"]
        .mean()
    )

    forecast_change = (
        (
            predicted_load
            -
            latest_load
        )
        /
        latest_load
    ) * 100

    baseline_difference = (
        (
            predicted_load
            -
            baseline
        )
        /
        baseline
    ) * 100

    forecast_rows.append(
        {
            "Athlete": athlete,
            "Latest_Load": latest_load,
            "Baseline_Load": baseline,
            "Forecast_Load": predicted_load,
            "Forecast_Change_%":
                forecast_change,
            "Forecast_vs_Baseline_%":
                baseline_difference
        }
    )


forecast = pd.DataFrame(
    forecast_rows
)


# ------------------------------------------
# Forecast Classification
# ------------------------------------------

def classify_forecast(row):

    change = abs(
        row["Forecast_Change_%"]
    )

    baseline_difference = abs(
        row["Forecast_vs_Baseline_%"]
    )

    if (
        change >= 20
        or
        baseline_difference >= 25
    ):
        return "REVIEW"

    elif (
        change >= 10
        or
        baseline_difference >= 15
    ):
        return "WATCH"

    else:
        return "STABLE"


forecast["Forecast_Status"] = (
    forecast.apply(
        classify_forecast,
        axis=1
    )
)


# ------------------------------------------
# Coaching Signal
# ------------------------------------------

def coaching_signal(status):

    if status == "REVIEW":

        return (
            "Review planned load and "
            "athlete context."
        )

    elif status == "WATCH":

        return (
            "Monitor upcoming load "
            "and athlete response."
        )

    else:

        return (
            "Forecast is within "
            "expected range."
        )


forecast["Coaching_Signal"] = (
    forecast["Forecast_Status"]
    .apply(coaching_signal)
)


# ------------------------------------------
# Display Forecast
# ------------------------------------------

print("\n" + "=" * 80)
print("TRAINING LOAD FORECAST")
print("=" * 80)

display_forecast = forecast.copy()

for column in [
    "Latest_Load",
    "Baseline_Load",
    "Forecast_Load"
]:

    display_forecast[column] = (
        display_forecast[column]
        .round(1)
    )

display_forecast[
    "Forecast_Change_%"
] = (
    display_forecast[
        "Forecast_Change_%"
    ].round(1)
)

display_forecast[
    "Forecast_vs_Baseline_%"
] = (
    display_forecast[
        "Forecast_vs_Baseline_%"
    ].round(1)
)

print(
    display_forecast.to_string(
        index=False
    )
)


# ------------------------------------------
# Latest Training Load
# ------------------------------------------

latest_data = (
    data.sort_values("Date")
    .groupby("Athlete")
    .tail(1)
)


print("\n" + "=" * 80)
print("LATEST ATHLETE LOAD")
print("=" * 80)

for _, row in latest_data.iterrows():

    print(
        f"{row['Athlete']:<10} "
        f"Latest Load: "
        f"{row['Training_Load']:>5.0f} AU | "
        f"3-Observation Average: "
        f"{row['Rolling_Load_3']:>5.1f} AU | "
        f"5-Observation Average: "
        f"{row['Rolling_Load_5']:>5.1f} AU"
    )


# ------------------------------------------
# Forecast Visualization
# ------------------------------------------

plt.figure(
    figsize=(12, 7)
)

for athlete in data["Athlete"].unique():

    athlete_data = data[
        data["Athlete"] == athlete
    ].sort_values("Date")

    plt.plot(
        athlete_data["Date"],
        athlete_data["Training_Load"],
        marker="o",
        label=f"{athlete} Actual"
    )

    athlete_forecast = forecast[
        forecast["Athlete"] == athlete
    ].iloc[0]

    next_date = (
        athlete_data["Date"].max()
        +
        pd.Timedelta(days=1)
    )

    plt.scatter(
        next_date,
        athlete_forecast["Forecast_Load"],
        marker="X",
        s=120,
        label=f"{athlete} Forecast"
    )

    plt.plot(
        [
            athlete_data["Date"].max(),
            next_date
        ],
        [
            athlete_data["Training_Load"].iloc[-1],
            athlete_forecast["Forecast_Load"]
        ],
        linestyle="--"
    )


plt.title(
    "Athlete Training Load Forecast"
)

plt.xlabel("Date")

plt.ylabel(
    "Training Load (AU)"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.savefig(
    "training_load_forecast.png",
    dpi=300
)

plt.show()


# ------------------------------------------
# Export Results
# ------------------------------------------

forecast.to_csv(
    "athlete_load_forecast_results.csv",
    index=False
)


# ------------------------------------------
# Final Output
# ------------------------------------------

print("\n" + "=" * 80)
print("FORECASTING ANALYSIS COMPLETE")
print("=" * 80)

print("Generated files:")

print(
    "1. athlete_load_forecast_results.csv"
)

print(
    "2. training_load_forecast.png"
)

print("\n" + "=" * 80)
print("OBSERVE • FORECAST • MONITOR • DECIDE")
print("=" * 80)