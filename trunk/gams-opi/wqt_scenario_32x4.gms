* Scenario 32x4 of model WQT
* python convert_to_gams.py 32 4 32x4/arrival.0 32x4/priorities.0 32x4/workload_high.0 32x4/cores_c8.22 32x4/scenario_c12_high.2 > gams_32x4.txt

$TITLE multicore_scheduling_min_wqt_s1
$Include wqt_model

SET t /1*32/;
SET m /1*4/;
SET c /1*12/;
PARAMETER m_cant_cores(m)
        /       1       12
                2       6
                3       4
                4       4 /;
PARAMETER m_cores(m, c)
        /       1       .1      1
                1       .2      1
                1       .3      1
                1       .4      1
                1       .5      1
                1       .6      1
                1       .7      1
                1       .8      1
                1       .9      1
                1       .10     1
                1       .11     1
                1       .12     1
                2       .1      1
                2       .2      1
                2       .3      1
                2       .4      1
                2       .5      1
                2       .6      1
                2       .7      0
                2       .8      0
                2       .9      0
                2       .10     0
                2       .11     0
                2       .12     0
                3       .1      1
                3       .2      1
                3       .3      1
                3       .4      1
                3       .5      0
                3       .6      0
                3       .7      0
                3       .8      0
                3       .9      0
                3       .10     0
                3       .11     0
                3       .12     0
                4       .1      1
                4       .2      1
                4       .3      1
                4       .4      1
                4       .5      0
                4       .6      0
                4       .7      0
                4       .8      0
                4       .9      0
                4       .10     0
                4       .11     0
                4       .12     0 /;
PARAMETER t_arrival(t)
        /       1       102.0
                2       104.0
                3       115.0
                4       121.0
                5       195.0
                6       212.0
                7       219.0
                8       232.0
                9       251.0
                10      262.0
                11      277.0
                12      284.0
                13      352.0
                14      362.0
                15      368.0
                16      390.0
                17      413.0
                18      421.0
                19      425.0
                20      474.0
                21      544.0
                22      567.0
                23      577.0
                24      584.0
                25      604.0
                26      630.0
                27      638.0
                28      646.0
                29      678.0
                30      686.0
                31      722.0
                32      746.0 /;
PARAMETER t_cores(t)
        /       1       8
                2       1
                3       1
                4       1
                5       1
                6       1
                7       1
                8       1
                9       1
                10      1
                11      1
                12      1
                13      1
                14      1
                15      1
                16      1
                17      1
                18      1
                19      1
                20      1
                21      1
                22      1
                23      1
                24      1
                25      8
                26      1
                27      1
                28      1
                29      1
                30      1
                31      2
                32      2 /;
PARAMETER t_priorities(t)
        /       1       3
                2       3
                3       3
                4       3
                5       3
                6       2
                7       4
                8       3
                9       2
                10      3
                11      3
                12      3
                13      3
                14      3
                15      3
                16      2
                17      2
                18      3
                19      3
                20      3
                21      1
                22      3
                23      4
                24      4
                25      4
                26      3
                27      3
                28      3
                29      3
                30      3
                31      3
                32      3 /;
PARAMETER etc(t,m)
        /       1       .1      11155.3327
                1       .2      9040.0421
                1       .3      7749.0257
                1       .4      11255.0432
                2       .1      23649.5811
                2       .2      19165.1129
                2       .3      16428.1262
                2       .4      23860.9700
                3       .1      1338.8525
                3       .2      1084.9774
                3       .3      930.0308
                3       .4      1350.8197
                4       .1      2832.3898
                4       .2      2295.3079
                4       .3      1967.5129
                4       .4      2857.7068
                5       .1      7572.0837
                5       .2      6136.2541
                5       .3      5259.9302
                5       .4      7639.7658
                6       .1      28150.0119
                6       .2      22812.1653
                6       .3      19554.3398
                6       .4      28401.6273
                7       .1      19531.7315
                7       .2      15828.0960
                7       .3      13567.6716
                7       .4      19706.3135
                8       .1      41220.4301
                8       .2      33404.1517
                8       .3      28633.6753
                8       .4      41588.8738
                9       .1      2279.3232
                9       .2      1847.1146
                9       .3      1583.3266
                9       .4      2299.6967
                10      .1      1238.7604
                10      .2      1003.8648
                10      .3      860.5020
                10      .4      1249.8329
                11      .1      2755.9187
                11      .2      2233.3374
                11      .3      1914.3925
                11      .4      2780.5522
                12      .1      42093.0512
                12      .2      34111.3051
                12      .3      29239.8394
                12      .4      42469.2947
                13      .1      20922.9101
                13      .2      16955.4772
                13      .3      14534.0505
                13      .4      21109.9269
                14      .1      6227.1157
                14      .2      5046.3209
                14      .3      4325.6514
                14      .4      6282.7760
                15      .1      23133.8827
                15      .2      18747.2019
                15      .3      16069.8974
                15      .4      23340.6620
                16      .1      28928.3970
                16      .2      23442.9519
                16      .3      20095.0432
                16      .4      29186.9699
                17      .1      2460.8555
                17      .2      1994.2245
                17      .3      1709.4275
                17      .4      2482.8516
                18      .1      5360.3285
                18      .2      4343.8951
                18      .3      3723.5396
                18      .4      5408.2411
                19      .1      4912.3382
                19      .2      3980.8534
                19      .3      3412.3442
                19      .4      4956.2466
                20      .1      6068.5712
                20      .2      4917.8398
                20      .3      4215.5188
                20      .4      6122.8143
                21      .1      31704.0772
                21      .2      25692.3036
                21      .3      22023.1630
                21      .4      31987.4602
                22      .1      4503.7844
                22      .2      3649.7702
                22      .3      3128.5433
                22      .4      4544.0409
                23      .1      8449.9827
                23      .2      6847.6846
                23      .3      5869.7607
                23      .4      8525.5119
                24      .1      10021.0467
                24      .2      8120.8411
                24      .3      6961.0966
                24      .4      10110.6186
                25      .1      17978.5943
                25      .2      14569.4669
                25      .3      12488.7884
                25      .4      18139.2937
                26      .1      42394.8647
                26      .2      34355.8883
                26      .3      29449.4935
                26      .4      42773.8059
                27      .1      15896.4738
                27      .2      12882.1611
                27      .3      11042.4483
                27      .4      16038.5625
                28      .1      9325.1369
                28      .2      7556.8907
                28      .3      6477.6845
                28      .4      9408.4884
                29      .1      5012.8681
                29      .2      4062.3207
                29      .3      3482.1771
                29      .4      5057.6750
                30      .1      8234.4908
                30      .2      6673.0546
                30      .3      5720.0698
                30      .4      8308.0938
                31      .1      3863.4857
                31      .2      3130.8859
                31      .3      2683.7613
                31      .4      3898.0190
                32      .1      37417.2094
                32      .2      30322.1033
                32      .3      25991.7769
                32      .4      37751.6586 /;
PARAMETER m_eidle(m)
        /       1       48.0
                2       61.0
                3       55.2
                4       92.7 /;
PARAMETER eec(t,m)
        /       1       .1      617261.7411
                1       .2      825657.1819
                1       .3      723759.0025
                1       .4      1521681.8457
                2       .1      163576.2695
                2       .2      218801.7055
                2       .3      191798.3730
                2       .4      403250.3930
                3       .1      9260.3967
                3       .2      12386.8247
                3       .3      10858.1093
                3       .4      22828.8529
                4       .1      19590.6961
                4       .2      26204.7652
                4       .3      22970.7136
                4       .4      48295.2444
                5       .1      52373.5788
                5       .2      70055.5673
                5       .3      61409.6851
                5       .4      129112.0422
                6       .1      194704.2486
                6       .2      260438.8876
                6       .3      228296.9175
                6       .4      479987.5008
                7       .1      135094.4764
                7       .2      180704.0956
                7       .3      158402.5657
                7       .4      333036.6982
                8       .1      285107.9748
                8       .2      381364.0654
                8       .3      334298.1586
                8       .4      702851.9676
                9       .1      15765.3191
                9       .2      21087.8920
                9       .3      18485.3375
                9       .4      38864.8741
                10      .1      8568.0928
                10      .2      11460.7903
                10      .3      10046.3610
                10      .4      21122.1762
                11      .1      19061.7713
                11      .2      25497.2686
                11      .3      22350.5325
                11      .4      46991.3320
                12      .1      291143.6039
                12      .2      389437.3999
                12      .3      341375.1255
                12      .4      717731.0808
                13      .1      144716.7945
                13      .2      193575.0311
                13      .3      169685.0394
                13      .4      356757.7647
                14      .1      43070.8838
                14      .2      57612.1638
                14      .3      50501.9796
                14      .4      106178.9149
                15      .1      160009.3552
                15      .2      214030.5554
                15      .3      187616.0526
                15      .4      394457.1884
                16      .1      200088.0791
                16      .2      267640.3680
                16      .3      234609.6298
                16      .4      493259.7913
                17      .1      17020.9172
                17      .2      22767.3961
                17      .3      19957.5662
                17      .4      41960.1912
                18      .1      37075.6053
                18      .2      49592.8028
                18      .3      43472.3251
                18      .4      91399.2748
                19      .1      33977.0060
                19      .2      45448.0768
                19      .3      39839.1191
                19      .4      83760.5667
                20      .1      41974.2839
                20      .2      56145.3378
                20      .3      49216.1815
                20      .4      103475.5624
                21      .1      219286.5341
                21      .2      293320.4663
                21      .3      257120.4282
                21      .4      540588.0775
                22      .1      31151.1755
                22      .2      41668.2099
                22      .3      36525.7430
                22      .4      76794.2917
                23      .1      58445.7139
                23      .2      78177.7327
                23      .3      68529.4564
                23      .4      144081.1504
                24      .1      69312.2396
                24      .2      92712.9363
                24      .3      81270.8031
                24      .4      170869.4541
                25      .1      994815.5493
                25      .2      1330677.9738
                25      .3      1166452.8380
                25      .4      2452432.5103
                26      .1      293231.1473
                26      .2      392229.7246
                26      .3      343822.8365
                26      .4      722877.3205
                27      .1      109950.6104
                27      .2      147071.3396
                27      .3      128920.5840
                27      .4      271051.7058
                28      .1      64498.8633
                28      .2      86274.5026
                28      .3      75626.9664
                28      .4      159003.4548
                29      .1      34672.3375
                29      .2      46378.1611
                29      .3      40654.4173
                29      .4      85474.7071
                30      .1      56955.2281
                30      .2      76184.0399
                30      .3      66781.8145
                30      .4      140406.7851
                31      .1      53444.8854
                31      .2      71488.5607
                31      .3      62665.8261
                31      .4      131753.0416
                32      .1      517604.7305
                32      .2      692354.6928
                32      .3      606907.9914
                32      .4      1276006.0591 /;

$Include wqt_solve
