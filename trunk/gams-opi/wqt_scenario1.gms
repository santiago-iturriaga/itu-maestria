* Scenario 8x2 of model WQT

* dimension 8x2
* scenario_c4_high.1
* arrival.0
* cores_c2.0
* priorities.0
* workload_high.0

$TITLE multicore_scheduling_min_wqt_s1
$Include wqt_model

SET t /1*8/;
SET m /1*2/;
SET c /1*4/;

PARAMETER m_cant_cores(m)
    /    1    2
        2    4 /;
PARAMETER m_cores(m, c)
    /    1    .1    1
        1    .2    1
        1    .3    0
        1    .4    0
        2    .1    1
        2    .2    1
        2    .3    1
        2    .4    1 /;
PARAMETER t_arrival(t)
    /    1    102
        2    104
        3    115
        4    121
        5    195
        6    212
        7    219
        8    232 /;
PARAMETER t_cores(t)
    /    1    1
        2    1
        3    1
        4    1
        5    1
        6    2
        7    1
        8    1 /;
PARAMETER t_priorities(t)
    /    1    3
        2    3
        3    3
        4    3
        5    3
        6    2
        7    4
        8    3 /;
PARAMETER etc(t,m)
    /    1    .1    4729.8822
        1    .2    4544.9179
        2    .1    3367.2307
        2    .2    3235.5535
        3    .1    14080.8667
        3    .2    13530.2278
        4    .1    1452.9241
        4    .2    1396.1068
        5    .1    24536.6753
        5    .2    23577.1571
        6    .1    18191.0048
        6    .2    17479.6370
        7    .1    14216.0811
        7    .2    13660.1545
        8    .1    1799.5884
        8    .2    1729.2147 /;
PARAMETER m_eidle(m)
    /    1    86.0
        2    63.5 /;
PARAMETER eec(t,m)
    /    1    .1    101692.4668
        1    .2    74423.0307
        2    .1    72395.4595
        2    .2    52982.1891
        3    .1    302738.6350
        3    .2    221557.4806
        4    .1    31237.8686
        4    .2    22861.2495
        5    .1    527538.5190
        5    .2    386075.9470
        6    .1    782213.2075
        6    .2    572458.1124
        7    .1    305645.7427
        7    .2    223685.0302
        8    .1    38691.1515
        8    .2    28315.8906 /;

$Include wqt_solve
