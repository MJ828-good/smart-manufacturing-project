import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyomo.environ as pyo
from pyomo.contrib import appsi


st.set_page_config(
    page_title="원예장비 총괄생산계획",
    page_icon="📊",
    layout="wide"
)


def solve_aggregate_plan(params):
    demand = params["demand"]
    n_months = len(demand)

    regular_wage = params["regular_wage"]
    overtime_wage = params["overtime_wage"]
    hire_cost = params["hire_cost"]
    layoff_cost = params["layoff_cost"]
    inventory_cost = params["inventory_cost"]
    shortage_cost = params["shortage_cost"]
    material_cost = params["material_cost"]
    outsourcing_cost = params["outsourcing_cost"]

    work_days = params["work_days"]
    work_hours_per_day = params["work_hours_per_day"]
    standard_time_per_unit = params["standard_time_per_unit"]
    max_overtime_per_worker = params["max_overtime_per_worker"]

    initial_workers = params["initial_workers"]
    initial_inventory = params["initial_inventory"]
    initial_shortage = params["initial_shortage"]
    final_min_inventory = params["final_min_inventory"]
    final_shortage = params["final_shortage"]

    integer_mode = params["integer_mode"]

    model = pyo.ConcreteModel()

    model.T = pyo.RangeSet(1, n_months)
    model.T0 = pyo.RangeSet(0, n_months)

    domain = pyo.NonNegativeIntegers if integer_mode else pyo.NonNegativeReals

    model.W = pyo.Var(model.T0, domain=domain)
    model.H = pyo.Var(model.T, domain=domain)
    model.L = pyo.Var(model.T, domain=domain)
    model.P = pyo.Var(model.T, domain=domain)
    model.I = pyo.Var(model.T0, domain=domain)
    model.S = pyo.Var(model.T0, domain=domain)
    model.C = pyo.Var(model.T, domain=domain)
    model.O = pyo.Var(model.T, domain=domain)

    monthly_regular_labor_cost = regular_wage * work_hours_per_day * work_days
    regular_capacity_per_worker = work_hours_per_day * work_days / standard_time_per_unit

    model.objective = pyo.Objective(
        expr=sum(
            monthly_regular_labor_cost * model.W[t]
            + overtime_wage * model.O[t]
            + hire_cost * model.H[t]
            + layoff_cost * model.L[t]
            + inventory_cost * model.I[t]
            + shortage_cost * model.S[t]
            + material_cost * model.P[t]
            + outsourcing_cost * model.C[t]
            for t in model.T
        ),
        sense=pyo.minimize
    )

    model.initial_workers_constraint = pyo.Constraint(
        expr=model.W[0] == initial_workers
    )

    model.initial_inventory_constraint = pyo.Constraint(
        expr=model.I[0] == initial_inventory
    )

    model.initial_shortage_constraint = pyo.Constraint(
        expr=model.S[0] == initial_shortage
    )

    def labor_balance_rule(m, t):
        return m.W[t] == m.W[t - 1] + m.H[t] - m.L[t]

    model.labor_balance = pyo.Constraint(model.T, rule=labor_balance_rule)

    def capacity_rule(m, t):
        return m.P[t] <= regular_capacity_per_worker * m.W[t] + m.O[t] / standard_time_per_unit

    model.capacity = pyo.Constraint(model.T, rule=capacity_rule)

    def inventory_balance_rule(m, t):
        return m.I[t] == m.I[t - 1] + m.P[t] + m.C[t] - demand[t - 1] - m.S[t - 1] + m.S[t]

    model.inventory_balance = pyo.Constraint(model.T, rule=inventory_balance_rule)

    def overtime_rule(m, t):
        return m.O[t] <= max_overtime_per_worker * m.W[t]

    model.overtime_limit = pyo.Constraint(model.T, rule=overtime_rule)

    model.final_inventory_constraint = pyo.Constraint(
        expr=model.I[n_months] >= final_min_inventory
    )

    model.final_shortage_constraint = pyo.Constraint(
        expr=model.S[n_months] == final_shortage
    )

    solver = appsi.solvers.Highs()

    if not bool(solver.available()):
        raise RuntimeError(
            "HiGHS solver를 사용할 수 없습니다. requirements.txt에 highspy가 있는지 확인하세요."
        )

    result = solver.solve(model)

    if result.termination_condition != appsi.base.TerminationCondition.optimal:
        raise RuntimeError(
            f"최적해를 찾지 못했습니다. 상태: {result.termination_condition}"
        )

    solver.load_vars()

    def val(x):
        return round(float(pyo.value(x)), 4)

    rows = []

    for t in range(1, n_months + 1):
        workers = val(model.W[t])
        overtime_hours = val(model.O[t])
        regular_capacity = regular_capacity_per_worker * workers
        overtime_capacity = overtime_hours / standard_time_per_unit
        total_capacity = regular_capacity + overtime_capacity

        rows.append({
            "월": f"{t}월",
            "수요": demand[t - 1],
            "월말 종업원": workers,
            "고용": val(model.H[t]),
            "해고": val(model.L[t]),
            "생산량": val(model.P[t]),
            "월말 재고": val(model.I[t]),
            "월말 부족재고": val(model.S[t]),
            "외주": val(model.C[t]),
            "초과근무시간": overtime_hours,
            "정규 생산가능량": round(regular_capacity, 4),
            "초과 생산가능량": round(overtime_capacity, 4),
            "총 생산가능량": round(total_capacity, 4)
        })

    result_df = pd.DataFrame(rows)

    cost_rows = [
        {
            "비용항목": "정규근무 노동비",
            "금액(천원)": sum(monthly_regular_labor_cost * val(model.W[t]) for t in range(1, n_months + 1))
        },
        {
            "비용항목": "초과근무 노동비",
            "금액(천원)": sum(overtime_wage * val(model.O[t]) for t in range(1, n_months + 1))
        },
        {
            "비용항목": "고용비",
            "금액(천원)": sum(hire_cost * val(model.H[t]) for t in range(1, n_months + 1))
        },
        {
            "비용항목": "해고비",
            "금액(천원)": sum(layoff_cost * val(model.L[t]) for t in range(1, n_months + 1))
        },
        {
            "비용항목": "재고유지비",
            "금액(천원)": sum(inventory_cost * val(model.I[t]) for t in range(1, n_months + 1))
        },
        {
            "비용항목": "부족재고비",
            "금액(천원)": sum(shortage_cost * val(model.S[t]) for t in range(1, n_months + 1))
        },
        {
            "비용항목": "재료비",
            "금액(천원)": sum(material_cost * val(model.P[t]) for t in range(1, n_months + 1))
        },
        {
            "비용항목": "외주비",
            "금액(천원)": sum(outsourcing_cost * val(model.C[t]) for t in range(1, n_months + 1))
        }
    ]

    cost_df = pd.DataFrame(cost_rows)
    total_cost = round(float(pyo.value(model.objective)), 4)

    return result_df, cost_df, total_cost


st.title("원예장비 제조업체 총괄생산계획 최적화 웹앱")

st.write(
    "월별 수요와 비용 파라미터를 입력하면 Pyomo 최적화 모델이 "
    "총비용이 최소가 되는 생산계획을 계산하고, 결과를 대시보드로 시각화합니다."
)

with st.expander("모델 설명 보기"):
    st.write(
        """
        이 모델은 월별 수요를 만족시키면서 생산량, 재고, 부족재고, 고용, 해고, 외주, 초과근무를 결정합니다.

        목적은 총비용 최소화입니다.

        총비용 = 정규근무 노동비 + 초과근무 노동비 + 고용비 + 해고비 + 재고유지비 + 부족재고비 + 재료비 + 외주비

        주요 제약조건은 다음과 같습니다.

        1. 인력 균형: 현재 종업원 = 전월 종업원 + 고용 - 해고
        2. 생산능력: 생산량 ≤ 정규 생산가능량 + 초과근무 생산가능량
        3. 재고 균형: 현재 재고 = 전월 재고 + 생산 + 외주 - 수요 - 전월 부족재고 + 현재 부족재고
        4. 초과근무 한도: 총 초과근무시간 ≤ 종업원 수 × 1인당 최대 초과근무시간
        5. 마지막 달 재고는 최소 재고 이상, 마지막 달 부족재고는 0
        """
    )

st.sidebar.header("1. 기본 설정")

n_months = st.sidebar.number_input(
    "계획 기간(개월)",
    min_value=1,
    max_value=12,
    value=6,
    step=1
)

default_demand = [1600, 3000, 3200, 3800, 2200, 2200]

st.sidebar.header("2. 월별 수요")

demand = []
for i in range(n_months):
    default_value = default_demand[i] if i < len(default_demand) else 2000
    value = st.sidebar.number_input(
        f"{i + 1}월 수요",
        min_value=0,
        value=default_value,
        step=100
    )
    demand.append(value)

st.sidebar.header("3. 초기/마지막 조건")

initial_workers = st.sidebar.number_input("초기 종업원 수", min_value=0, value=80, step=1)
initial_inventory = st.sidebar.number_input("초기 재고", min_value=0, value=1000, step=100)
initial_shortage = st.sidebar.number_input("초기 부족재고", min_value=0, value=0, step=10)
final_min_inventory = st.sidebar.number_input("마지막 달 최소 재고", min_value=0, value=500, step=100)
final_shortage = st.sidebar.number_input("마지막 달 부족재고", min_value=0, value=0, step=10)

st.sidebar.header("4. 생산능력 조건")

work_days = st.sidebar.number_input("월 작업일수", min_value=1, value=20, step=1)
work_hours_per_day = st.sidebar.number_input("하루 작업시간", min_value=1.0, value=8.0, step=1.0)
standard_time_per_unit = st.sidebar.number_input("제품 1개당 작업시간", min_value=0.1, value=4.0, step=0.1)
max_overtime_per_worker = st.sidebar.number_input("1인당 월 최대 초과근무시간", min_value=0.0, value=10.0, step=1.0)

st.sidebar.header("5. 비용 조건")

regular_wage = st.sidebar.number_input("정규근무 임금(천원/시간)", min_value=0.0, value=4.0, step=0.5)
overtime_wage = st.sidebar.number_input("초과근무 임금(천원/시간)", min_value=0.0, value=6.0, step=0.5)
hire_cost = st.sidebar.number_input("고용비용(천원/명)", min_value=0.0, value=300.0, step=50.0)
layoff_cost = st.sidebar.number_input("해고비용(천원/명)", min_value=0.0, value=500.0, step=50.0)
inventory_cost = st.sidebar.number_input("재고유지비(천원/개/월)", min_value=0.0, value=2.0, step=0.5)
shortage_cost = st.sidebar.number_input("부족재고비(천원/개/월)", min_value=0.0, value=5.0, step=0.5)
material_cost = st.sidebar.number_input("재료비(천원/개)", min_value=0.0, value=10.0, step=1.0)
outsourcing_cost = st.sidebar.number_input("외주비용(천원/개)", min_value=0.0, value=30.0, step=1.0)

st.sidebar.header("6. 최적화 설정")

integer_mode = st.sidebar.checkbox("정수계획으로 계산", value=True)

params = {
    "demand": demand,
    "regular_wage": regular_wage,
    "overtime_wage": overtime_wage,
    "hire_cost": hire_cost,
    "layoff_cost": layoff_cost,
    "inventory_cost": inventory_cost,
    "shortage_cost": shortage_cost,
    "material_cost": material_cost,
    "outsourcing_cost": outsourcing_cost,
    "work_days": work_days,
    "work_hours_per_day": work_hours_per_day,
    "standard_time_per_unit": standard_time_per_unit,
    "max_overtime_per_worker": max_overtime_per_worker,
    "initial_workers": initial_workers,
    "initial_inventory": initial_inventory,
    "initial_shortage": initial_shortage,
    "final_min_inventory": final_min_inventory,
    "final_shortage": final_shortage,
    "integer_mode": integer_mode
}

try:
    result_df, cost_df, total_cost = solve_aggregate_plan(params)

    st.success("최적 생산계획 계산 완료")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("총비용", f"{total_cost:,.0f} 천원")
    col2.metric("총 생산량", f"{result_df['생산량'].sum():,.0f} 개")
    col3.metric("총 외주량", f"{result_df['외주'].sum():,.0f} 개")
    col4.metric("마지막 달 재고", f"{result_df['월말 재고'].iloc[-1]:,.0f} 개")

    st.subheader("월별 총괄생산계획 결과표")
    st.dataframe(result_df, use_container_width=True)

    st.subheader("비용 구성표")
    st.dataframe(cost_df, use_container_width=True)

    st.subheader("대시보드")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=result_df["월"],
        y=result_df["수요"],
        mode="lines+markers",
        name="수요"
    ))
    fig1.add_trace(go.Scatter(
        x=result_df["월"],
        y=result_df["생산량"],
        mode="lines+markers",
        name="생산량"
    ))
    fig1.update_layout(
        title="수요와 생산량 비교",
        xaxis_title="월",
        yaxis_title="수량"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(
        result_df,
        x="월",
        y=["월말 재고", "월말 부족재고"],
        barmode="group",
        title="월별 재고와 부족재고"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(
        result_df,
        x="월",
        y=["생산량", "총 생산가능량"],
        barmode="group",
        title="생산량과 생산가능량 비교"
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.line(
        result_df,
        x="월",
        y="월말 종업원",
        markers=True,
        title="월별 종업원 수 변화"
    )
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.pie(
        cost_df,
        names="비용항목",
        values="금액(천원)",
        title="총비용 구성 비율"
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("계획 해석")

    max_shortage = result_df["월말 부족재고"].max()
    total_overtime = result_df["초과근무시간"].sum()
    total_outsourcing = result_df["외주"].sum()

    if max_shortage == 0:
        st.write("부족재고가 발생하지 않아 수요 충족 측면에서는 안정적인 계획입니다.")
    else:
        st.write("일부 월에서 부족재고가 발생하므로 수요 충족 측면에서 추가 검토가 필요합니다.")

    if total_overtime == 0:
        st.write("초과근무를 사용하지 않는 계획입니다.")
    else:
        st.write(f"총 초과근무시간은 {total_overtime:,.0f}시간입니다. 초과근무 의존도가 적절한지 확인해야 합니다.")

    if total_outsourcing == 0:
        st.write("외주 생산을 사용하지 않는 계획입니다.")
    else:
        st.write(f"총 외주량은 {total_outsourcing:,.0f}개입니다. 외주비가 총비용에 미치는 영향을 확인해야 합니다.")

except Exception as e:
    st.error("최적화 계산 중 오류가 발생했습니다.")
    st.exception(e)