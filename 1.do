clear all

* 以全部字符串形式导入（firstrow 使得首行变变量名）
import excel "3.2data_with_PeerDigital.xlsx", sheet("Sheet1") firstrow allstring clear
* 检查一下变量类型
describe
* 将以下变量从字符串转为数值
* force 选项会把空格、非数字字符变成系统缺失值 '.'，不会变成0
destring Digital MD_Digital PeerDigital Subsidy Analyst ///
         Size Lev ROA Growth PPE Age Board Dual Competition Synch ///
         Big4 Top10 SA, replace force
save "数据_数值版.dta", replace
clear all
use "数据_数值版.dta", clear
describe ROA
sum ROA, detail


clear all
* 1. 从源头开始
use "数据_数值版.dta", clear

* 2. 手工精准切除那两个你确认过的“妖怪” (ROA > 10)
* 这行命令会删掉你说的那两个 ROA 为 10 和 20 的观测
drop if ROA > 10

* 3. 确认一下，ROA 的最大值现在应该回归正常范围
sum ROA, detail

* 4. 对所有连续变量进行标准的 1% 和 99% 缩尾
* 这会处理掉其他所有统计上的微弱极值
local cont_vars "Digital PeerDigital Subsidy Analyst Size Lev ROA Growth PPE Age Board Competition Synch"
winsor2 `cont_vars', replace cuts(1 99)

* 5. 最终验证
sum `cont_vars', detail

* 6. 保存最终数据 
save "数据_最终分析2.dta", replace



clear all
use "数据_最终分析2.dta", clear
* 生成描述性统计表
sum Digital PeerDigital Subsidy Analyst Size Lev ROA Growth PPE Age Board Dual Competition Synch Big4 SA, detail
tabstat Digital PeerDigital Subsidy Analyst Size Lev ROA Growth PPE Age Board Dual Competition, ///
        stats(N mean sd min p50 max) columns(statistics) format(%9.3f)

// 回归分析：Y为Digital，X为PeerDigital
regress Digital PeerDigital
estimates store m1

* 加载最终数据
use "数据_最终分析2.dta", clear
* 先 destring 标识符
destring code year, replace

* 设定面板结构
xtset code year

* ==========================
* 模型 (1): 只有控制变量，没有固定效应
* ==========================
reg Digital PeerDigital Size Lev ROA Growth PPE Age Board Dual Competition, robust
estimates store m2

* ==========================
* 模型 (3): 控制变量 + 年份固定效应 + 企业个体固定效应
* ==========================
xtreg Digital PeerDigital Size Lev ROA Growth PPE Age Board Dual Competition i.year, fe robust
estimates store m3

* ==========================
* 汇总输出三个模型的结果对比
* ==========================
estimates table m1 m2 m3, star stats(N r2 r2_w) b(%9.4f)



*关于ROA系数为负的检验
* ===========================================
* 检验一：数字化是否侵蚀未来盈利能力？
* ===========================================
use "数据_最终分析2.dta", clear
destring code year, replace
xtset code year

* 生成未来一期的 ROA
gen F_ROA = F.ROA

* 生成未来二期的 ROA (看效果是否持续)
gen F2_ROA = F2.ROA

* 回归1：当期Digital -> 下一期ROA
xtreg F_ROA Digital Size Lev Growth PPE Age Board Dual Competition i.year, fe robust
estimates store lag1

* 回归2：当期Digital -> 下两期ROA
xtreg F2_ROA Digital Size Lev Growth PPE Age Board Dual Competition i.year, fe robust
estimates store lag2

estimates table lag1 lag2, star stats(N r2_w) b(%9.4f)

* ===========================================
* 检验二：“穷则思变” —— 过去的盈利压力是否驱动当期的数字化？
* ===========================================
use "数据_最终分析2.dta", clear
destring code year, replace
xtset code year

* 生成滞后一期的 ROA
gen L_ROA = L.ROA

* 用滞后一期ROA替换当期ROA，跑你的模型(3)
xtreg Digital PeerDigital Size Lev L_ROA Growth PPE Age Board Dual Competition i.year, fe robust
estimates store poor_drive

estimates table poor_drive, star stats(N r2_w) b(%9.4f)


* ===========================================
* 检验三：分样本 —— “穷则思变”只在低盈利组显著吗？
* ===========================================
use "数据_最终分析2.dta", clear
destring code year, replace
xtset code year

* 计算每家企业自身的平均ROA
bysort code: egen mean_ROA = mean(ROA)

* 按平均ROA的中位数分为两组
sum mean_ROA, detail
gen high_profit = (mean_ROA > r(p50))

* 低盈利组回归
xtreg Digital PeerDigital Size Lev ROA Growth PPE Age Board Dual Competition i.year if high_profit == 0, fe robust
estimates store low_profit

* 高盈利组回归
xtreg Digital PeerDigital Size Lev ROA Growth PPE Age Board Dual Competition i.year if high_profit == 1, fe robust
estimates store high_profit

estimates table low_profit high_profit, star stats(N r2_w) b(%9.4f)

*再次对检验三进行检验
* 1. 重新跑两个非稳健的FE模型
xtreg Digital PeerDigital Size Lev ROA Growth PPE Age Board Dual Competition i.year if high_profit == 0, fe
estimates store low

xtreg Digital PeerDigital Size Lev ROA Growth PPE Age Board Dual Competition i.year if high_profit == 1, fe
estimates store high

* 2. 全样本加入交乘项
* 确保数据已加载、xtset已设定
use "数据_最终分析2.dta", clear
destring code year, replace
xtset code year

* 生成高盈利组虚拟变量（如果还没建）
bysort code: egen mean_ROA = mean(ROA)
sum mean_ROA, detail
gen high_profit = (mean_ROA > r(p50))

* 生成交乘项
gen ROA_high = ROA * high_profit

* 全样本回归 + 交乘项
xtreg Digital PeerDigital Size Lev ROA Growth PPE Age Board Dual Competition ///
      high_profit ROA_high i.year, fe robust



clear all
* ============================================
* 5.3 内生性处理与稳健性检验（完整重跑）
* ============================================

* --- 0. 加载数据 & 设定面板 ---
use "数据_最终分析2.dta", clear
destring code year, replace
xtset code year

* --- 1. 生成所需变量 ---

* 1.1 滞后一期 PeerDigital
gen L_PeerDigital = L.PeerDigital

* 1.2 生成取对数的 MD_Digital
gen MD_Digital_log = ln(1 + MD_Digital)
label var MD_Digital_log "文本分析数字化词频（对数）"

* 1.3 生成 MD_PeerDigital_log（leave-one-out 行业均值）
* 确认行业变量名（请根据实际变量名调整，假设为 Industry）
bysort Industry year: egen sum_MD = total(MD_Digital_log) if !missing(MD_Digital_log)
bysort Industry year: egen count_MD = count(MD_Digital_log) if !missing(MD_Digital_log)
gen MD_PeerDigital_log = (sum_MD - MD_Digital_log) / (count_MD - 1) ///
                         if !missing(MD_Digital_log) & count_MD > 1
drop sum_MD count_MD
label var MD_PeerDigital_log "同行文本分析数字化词频（对数，leave-one-out）"

* 1.4 行业中位数 PeerDigital
bysort Industry year: egen PeerDigital_median = median(Digital)
label var PeerDigital_median "行业同行数字化中位数"

* --- 2. 模型估计 ---

* (1) 滞后一期 PeerDigital
xtreg Digital L_PeerDigital Size Lev ROA Growth PPE Age Board Dual Competition ///
      i.year, fe robust
estimates store robust1

* (2) 替换被解释变量为 MD_Digital_log，核心解释变量为 MD_PeerDigital_log
xtreg MD_Digital_log MD_PeerDigital_log Size Lev ROA Growth PPE Age Board Dual Competition ///
      i.year, fe robust
estimates store robust2

* (3) 替换核心解释变量为行业中位数
xtreg Digital PeerDigital_median Size Lev ROA Growth PPE Age Board Dual Competition ///
      i.year, fe robust
estimates store robust3

* (4) 剔除疫情年份（2020–2022）
xtreg Digital PeerDigital Size Lev ROA Growth PPE Age Board Dual Competition ///
      i.year if year < 2020 | year > 2022, fe robust
estimates store robust4

* --- 3. 汇总输出（只显示核心变量） ---
estimates table robust1 robust2 robust3 robust4, ///
      star stats(N r2_w) b(%9.4f) ///
      keep(L_PeerDigital PeerDigital MD_PeerDigital_log PeerDigital_median)

* --- 4. 保存最终数据 ---
save "数据_最终分析3.dta", replace