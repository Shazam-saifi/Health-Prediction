from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Streamlit is not installed. Run `pip install -r requirements.txt` to launch the app."
    ) from exc

from health_predict_ai.predict import predict_risk


ARTIFACTS_REQUIRED = [
    Path("artifacts/heart_disease_bundle.joblib"),
    Path("artifacts/diabetes_bundle.joblib"),
    Path("artifacts/model.pkl"),
]
USERS_PATH = Path("data/users.json")


HEART_DEFAULTS = {
    "age": 54,
    "sex": 1,
    "chest_pain_type": 2,
    "resting_bp": 138,
    "cholesterol": 245,
    "fasting_blood_sugar": 0,
    "resting_ecg": 1,
    "max_heart_rate": 150,
    "exercise_angina": 0,
    "st_depression": 1.2,
}

DIABETES_DEFAULTS = {
    "age": 46,
    "sex": 0,
    "bmi": 31.8,
    "glucose": 145,
    "blood_pressure": 84,
    "insulin": 115,
    "pregnancies": 2,
    "physical_activity": 1,
    "smoker": 0,
    "family_history": 1,
}

SEX_OPTIONS = {"Female": 0, "Male": 1}
YES_NO_OPTIONS = {"No": 0, "Yes": 1}

HEART_REFERENCE_RANGES = pd.DataFrame(
    [
        {"Reading": "Age", "Minimum": "29 years", "Average": "54 years", "High": "70+ years"},
        {"Reading": "Resting BP", "Minimum": "90 mmHg", "Average": "120-129 mmHg", "High": "140+ mmHg"},
        {"Reading": "Cholesterol", "Minimum": "120 mg/dL", "Average": "180-200 mg/dL", "High": "240+ mg/dL"},
        {"Reading": "Max Heart Rate", "Minimum": "70 bpm", "Average": "140-160 bpm", "High": "180+ bpm"},
        {"Reading": "ST Depression", "Minimum": "0.0", "Average": "0.5-1.0", "High": "2.0+"},
        {"Reading": "Fasting Blood Sugar", "Minimum": "0", "Average": "0", "High": "1"},
    ]
)

DIABETES_REFERENCE_RANGES = pd.DataFrame(
    [
        {"Reading": "Sex", "Minimum": "Male/Female", "Average": "Adult", "High": "Used as context"},
        {"Reading": "Age", "Minimum": "21 years", "Average": "45 years", "High": "65+ years"},
        {"Reading": "BMI", "Minimum": "16", "Average": "18.5-24.9", "High": "30+"},
        {"Reading": "Glucose", "Minimum": "65 mg/dL", "Average": "70-99 mg/dL", "High": "140+ mg/dL"},
        {"Reading": "Blood Pressure", "Minimum": "40 mmHg", "Average": "120/80 equivalent", "High": "140/90+ equivalent"},
        {"Reading": "Insulin", "Minimum": "15 uU/mL", "Average": "16-166 uU/mL", "High": "200+ uU/mL"},
        {"Reading": "Pregnancies", "Minimum": "0", "Average": "1-2", "High": "5+"},
    ]
)


def _ensure_storage() -> None:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_PATH.exists():
        USERS_PATH.write_text("[]")


def _load_users() -> list[dict[str, str]]:
    _ensure_storage()
    return json.loads(USERS_PATH.read_text())


def _save_users(users: list[dict[str, str]]) -> None:
    _ensure_storage()
    USERS_PATH.write_text(json.dumps(users, indent=2))


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _register_user(full_name: str, email: str, password: str) -> tuple[bool, str]:
    cleaned_name = full_name.strip()
    cleaned_email = email.strip().lower()

    if len(cleaned_name) < 3:
        return False, "Full name must be at least 3 characters."
    if "@" not in cleaned_email or "." not in cleaned_email:
        return False, "Enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    users = _load_users()
    if any(user["email"] == cleaned_email for user in users):
        return False, "An account with this email already exists."

    users.append(
        {
            "full_name": cleaned_name,
            "email": cleaned_email,
            "password_hash": _hash_password(password),
        }
    )
    _save_users(users)
    return True, "Registration successful. You can now sign in."


def _authenticate_user(email: str, password: str) -> dict[str, str] | None:
    cleaned_email = email.strip().lower()
    password_hash = _hash_password(password)
    for user in _load_users():
        if user["email"] == cleaned_email and user["password_hash"] == password_hash:
            return user
    return None


def _render_auth_tabs(prefix: str) -> None:
    sign_in_tab, register_tab = st.tabs(["Sign In", "Register"])

    with sign_in_tab:
        with st.form(f"{prefix}_sign_in_form"):
            login_email = st.text_input("Email", key=f"{prefix}_login_email")
            login_password = st.text_input("Password", type="password", key=f"{prefix}_login_password")
            if st.form_submit_button("Sign In", width="stretch"):
                user = _authenticate_user(login_email, login_password)
                if user:
                    st.session_state.current_user = user
                    st.success("Signed in successfully.")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with register_tab:
        with st.form(f"{prefix}_register_form", clear_on_submit=True):
            full_name = st.text_input("Full Name", key=f"{prefix}_register_name")
            register_email = st.text_input("Email", key=f"{prefix}_register_email")
            register_password = st.text_input("Password", type="password", key=f"{prefix}_register_password")
            if st.form_submit_button("Create Account", width="stretch"):
                success, message = _register_user(full_name, register_email, register_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)


def _render_auth_panel() -> None:
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    with st.sidebar:
        st.subheader("User Access")
        st.caption("Manage your session here.")
        st.success(f"Signed in as {st.session_state.current_user['full_name']}")
        st.caption(st.session_state.current_user["email"])
        if st.button("Log Out", width="stretch"):
            st.session_state.current_user = None
            st.rerun()


def _render_auth_page() -> None:
    st.markdown(
        """
        <div style="
            max-width: 760px;
            margin: 20px auto 28px auto;
            background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
            border: 1px solid #dbe7f3;
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
        ">
            <div style="
                display: inline-block;
                padding: 6px 12px;
                border-radius: 999px;
                background: #e0ecff;
                color: #1d4ed8;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 12px;
            ">Secure Access</div>
            <div style="
                font-size: 40px;
                font-weight: 800;
                color: #0f172a;
                line-height: 1.05;
                margin-bottom: 12px;
            ">Welcome to HealthPredict AI</div>
            <div style="
                font-size: 20px;
                line-height: 1.6;
                color: #334155;
            ">
                Sign in or create an account first. After login, you will get access to the heart disease
                and diabetes health screening tools.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    auth_col_left, auth_col_center, auth_col_right = st.columns([1, 1.2, 1])
    with auth_col_center:
        _render_auth_tabs("page")


def _check_artifacts() -> None:
    missing = [str(path) for path in ARTIFACTS_REQUIRED if not path.exists()]
    if missing:
        st.error("Model artifacts are missing. Run `python3 -m health_predict_ai.train` first.")
        st.code("\n".join(missing))
        st.stop()


def _number_input_fields(defaults: dict[str, float | int]) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    for key, value in defaults.items():
        if key == "sex":
            default_label = next(label for label, encoded in SEX_OPTIONS.items() if encoded == value)
            payload[key] = SEX_OPTIONS[
                st.selectbox("Sex", options=list(SEX_OPTIONS.keys()), index=list(SEX_OPTIONS.keys()).index(default_label))
            ]
            continue
        if key == "chest_pain_type":
            default_label = next(label for label, encoded in YES_NO_OPTIONS.items() if encoded == min(int(value), 1))
            payload[key] = YES_NO_OPTIONS[
                st.selectbox(
                    "Chest Pain",
                    options=list(YES_NO_OPTIONS.keys()),
                    index=list(YES_NO_OPTIONS.keys()).index(default_label),
                )
            ]
            continue

        label = key.replace("_", " ").title()
        if isinstance(value, int):
            payload[key] = st.number_input(label, value=int(value), step=1)
        else:
            payload[key] = st.number_input(label, value=float(value), step=0.1, format="%.2f")
    return payload


def _diabetes_input_fields(defaults: dict[str, float | int]) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}

    sex_label = next(label for label, encoded in SEX_OPTIONS.items() if encoded == defaults["sex"])
    selected_sex = st.radio(
        "Sex",
        options=list(SEX_OPTIONS.keys()),
        index=list(SEX_OPTIONS.keys()).index(sex_label),
        horizontal=True,
        key="diabetes_sex",
    )
    payload["sex"] = SEX_OPTIONS[selected_sex]

    payload["age"] = st.number_input("Age", value=int(defaults["age"]), step=1, key="diabetes_age")
    payload["bmi"] = st.number_input("Bmi", value=float(defaults["bmi"]), step=0.1, format="%.2f", key="diabetes_bmi")
    payload["glucose"] = st.number_input(
        "Glucose", value=int(defaults["glucose"]), step=1, key="diabetes_glucose"
    )
    payload["blood_pressure"] = st.number_input(
        "Blood Pressure", value=int(defaults["blood_pressure"]), step=1, key="diabetes_blood_pressure"
    )
    payload["insulin"] = st.number_input(
        "Insulin", value=int(defaults["insulin"]), step=1, key="diabetes_insulin"
    )

    if selected_sex == "Female":
        payload["pregnancies"] = st.number_input(
            "Pregnancies",
            min_value=0,
            value=int(defaults["pregnancies"]),
            step=1,
            key="diabetes_pregnancies",
        )
    else:
        payload["pregnancies"] = 0
        st.caption("Pregnancies is automatically set to 0 for male users.")

    payload["physical_activity"] = st.number_input(
        "Physical Activity", value=int(defaults["physical_activity"]), step=1, key="diabetes_physical_activity"
    )
    payload["smoker"] = st.number_input("Smoker", value=int(defaults["smoker"]), step=1, key="diabetes_smoker")
    payload["family_history"] = st.number_input(
        "Family History", value=int(defaults["family_history"]), step=1, key="diabetes_family_history"
    )

    return payload


def _render_reference_ranges(title: str, ranges_df: pd.DataFrame) -> None:
    st.markdown(
        f"""
        <div style="
            background: #ffffff;
            border: 1px solid #dbe7f3;
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 12px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
        ">
            <div style="font-size: 1.02rem; font-weight: 800; color: #0f172a; margin-bottom: 0.35rem;">{title}</div>
            <div style="font-size: 0.95rem; line-height: 1.6; color: #475569;">
                Use these guide values to understand what low, average, and high readings look like.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(ranges_df, width="stretch", hide_index=True)


def _render_input_card(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div style="
            background: #ffffff;
            border: 1px solid #dbe7f3;
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 12px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
        ">
            <div style="font-size: 1.02rem; font-weight: 800; color: #0f172a; margin-bottom: 0.35rem;">{title}</div>
            <div style="font-size: 0.95rem; line-height: 1.6; color: #475569;">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    hero_left, hero_right = st.columns([1.55, 0.85], gap="large")

    with hero_left:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #f8fbff 0%, #eef6ff 100%);
                border: 1px solid #dbe7f3;
                border-radius: 26px;
                padding: 30px 32px;
                box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
                min-height: 250px;
            ">
                <div style="
                    display: inline-block;
                    padding: 6px 12px;
                    border-radius: 999px;
                    background: #e0ecff;
                    color: #1d4ed8;
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 0.08em;
                    text-transform: uppercase;
                    margin-bottom: 16px;
                ">AI Health Screening</div>
                <div style="
                    font-size: 46px;
                    line-height: 1.02;
                    font-weight: 800;
                    color: #0f172a;
                    margin-bottom: 14px;
                ">HealthPredict AI</div>
                <div style="
                    font-size: 21px;
                    line-height: 1.7;
                    color: #334155;
                    max-width: 900px;
                ">
                    Explore heart disease and diabetes risk through a guided clinical-style dashboard.
                    Compare readings with reference values, enter patient data clearly, and review the strongest model factors instantly.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_right:
        st.markdown(
            """
            <div style="
                background: #ffffff;
                border: 1px solid #dbe7f3;
                border-radius: 26px;
                padding: 28px;
                box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
                min-height: 250px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <div style="font-size: 13px; color: #64748b; margin-bottom: 10px; font-weight: 600;">Platform Focus</div>
                <div style="font-size: 34px; line-height: 1.2; font-weight: 800; color: #0f172a; margin-bottom: 14px;">
                    Early Risk<br>Screening
                </div>
                <div style="font-size: 17px; line-height: 1.7; color: #475569;">
                    Designed for quick health checks with a simple workflow from login to prediction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    info_col_1, info_col_2, info_col_3 = st.columns(3, gap="large")
    with info_col_1:
        st.markdown(
            """
            <div style="
                background: #ffffff;
                border: 1px solid #dbe7f3;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
            ">
                <div style="font-size: 13px; color: #64748b; margin-bottom: 8px;">Models</div>
                <div style="font-size: 24px; font-weight: 800; color: #0f172a;">Heart + Diabetes</div>
                <div style="font-size: 15px; line-height: 1.6; color: #475569; margin-top: 10px;">
                    Dual prediction flows for two major chronic disease screening tasks.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_col_2:
        st.markdown(
            """
            <div style="
                background: #ffffff;
                border: 1px solid #dbe7f3;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
            ">
                <div style="font-size: 13px; color: #64748b; margin-bottom: 8px;">Prediction Style</div>
                <div style="font-size: 24px; font-weight: 800; color: #0f172a;">Real-time Risk Scoring</div>
                <div style="font-size: 15px; line-height: 1.6; color: #475569; margin-top: 10px;">
                    Enter values and receive an immediate machine learning risk estimate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_col_3:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
                border: 1px solid #dbe7f3;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
            ">
                <div style="font-size: 13px; color: #64748b; margin-bottom: 8px;">Interface</div>
                <div style="font-size: 24px; font-weight: 800; color: #0f172a;">White, Guided, Interactive</div>
                <div style="font-size: 15px; line-height: 1.6; color: #475569; margin-top: 10px;">
                    Built to keep the workflow clear from registration to final prediction review.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_result(result: dict[str, object]) -> None:
    pill_background = "#fff1f2" if result["risk_label"] == "High Risk" else "#ecfdf5"
    pill_color = "#be123c" if result["risk_label"] == "High Risk" else "#047857"
    st.markdown(
        f"""
        <div style="
            background: #ffffff;
            border: 1px solid #dbe7f3;
            border-radius: 18px;
            padding: 18px 20px;
            margin-top: 12px;
            margin-bottom: 12px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        ">
            <div style="
                display: inline-block;
                padding: 6px 12px;
                border-radius: 999px;
                background: {pill_background};
                color: {pill_color};
                font-size: 0.82rem;
                font-weight: 800;
                margin-bottom: 10px;
            ">{result["risk_label"]}</div>
            <div style="font-size: 1.02rem; font-weight: 800; color: #0f172a; margin-bottom: 0.35rem;">Prediction Summary</div>
            <div style="font-size: 0.95rem; line-height: 1.6; color: #475569;">
                Review the overall model decision and the strongest contributing features below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.metric("Risk Probability", f"{result['risk_probability']:.2%}")
    st.dataframe(result["top_factors"], width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="HealthPredict AI", page_icon=":hospital:", layout="wide")

    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    if not st.session_state.current_user:
        _render_auth_page()
        return

    _render_auth_panel()
    _check_artifacts()

    _render_hero()
    st.caption(f"Welcome back, {st.session_state.current_user['full_name']}.")

    tab1, tab2 = st.tabs(["Heart Disease", "Diabetes"])

    with tab1:
        st.subheader("Heart Disease Risk Prediction")
        left_col, right_col = st.columns([1.2, 0.95], gap="large")
        with left_col:
            _render_reference_ranges("Heart Health Reference Readings", HEART_REFERENCE_RANGES)
        with right_col:
            _render_input_card(
                "Enter Patient Readings",
                "Adjust the values below to generate a heart disease risk estimate.",
            )
            heart_payload = _number_input_fields(HEART_DEFAULTS)
            if st.button("Predict Heart Disease Risk", width="stretch"):
                result = predict_risk("heart_disease", heart_payload)
                _render_result(result)

    with tab2:
        st.subheader("Diabetes Risk Prediction")
        left_col, right_col = st.columns([1.2, 0.95], gap="large")
        with left_col:
            _render_reference_ranges("Diabetes Reference Readings", DIABETES_REFERENCE_RANGES)
        with right_col:
            _render_input_card(
                "Enter Patient Readings",
                "Use the sex toggle and input values to estimate diabetes risk more naturally.",
            )
            diabetes_payload = _diabetes_input_fields(DIABETES_DEFAULTS)
            if st.button("Predict Diabetes Risk", width="stretch"):
                result = predict_risk("diabetes", diabetes_payload)
                _render_result(result)


if __name__ == "__main__":
    main()
