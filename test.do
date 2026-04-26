* 测试 Stata 扩展是否配置成功
clear all
set more off

* 1. 显示 Stata 系统路径（确认 Stata 内核已启动）
sysdir

* 2. 基本运算测试
display "2 + 3 = " 2+3
display "正常中文字符显示测试"

* 3. 创建一个小数据集并做简单统计
clear
input id age income
1 25 5000
2 30 6000
3 28 5500
4 35 7000
5 40 8000
end

list
summarize age income

* 4. 若一切正常，会看到上述命令的输出结果（无报错）
display "测试完成！"
import excel "D:\RESEARCH\keti\data\projecct1\2data.xlsx", firstrow clear 
browse
list in 1/10