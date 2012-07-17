<%
/**
 * Copyright (c) 2000-2011 Liferay, Inc. All rights reserved.
 *
 * This library is free software; you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation; either version 2.1 of the License, or (at your option)
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
 * details.
 */
%>

<%@ taglib uri="http://java.sun.com/portlet_2_0" prefix="portlet" %>

<portlet:defineObjects />
<%  
  // Gets the current timestamp
  java.util.Date date = new java.util.Date();
%>

<div class="MyPortletWebapp">
    <!-- portlet content here -->
<span>
    <img width="155" height="37" src="<%=renderRequest.getContextPath()%>/images/logo.jpg">
    
    <span style="position:relative;left:8px;top:0px">
    <p style="background:#5EA3CE;width:440px; height:50px;position:relative;left:140px;top:-35px;">
        <span style="font-size:18.0pt;font-family:Arial;color:white;position:relative;left:10px;top:15px;">ME-MLS</span>
        <span style="font-family:Arial;color:white;font-weight:bold;position:relative;left:20px;top:15px;">Makespan-Energy Multithreading Local Search solver</span>
    </p>
    </span>
</span>

<div> </div>
    <span style="position:relative;left:4px;top:-35px;width:581px;height:4px">
    <img width="581" height="4" src="<%=renderRequest.getContextPath()%>/images/bar.gif" v:shapes="_x0000_s1028">
    </span>

<div> </div>

<span style="position:relative; left:0px;top:-10px">
        ME-MLS is an efficient multithreading local search algorithm for solving the multiobjective scheduling 
        problem in heterogeneous computing systems considering the makespan and energy consumption objectives. The
        proposed method follows a fully multiobjective approach using a Pareto-based dominance search executed in 
        parallel. The experimental analysis demonstrates that the new multithreading algorithm outperforms a set 
        of deterministic heuristics based on Min-Min. The new method is able to achieve significant improvements 
        in both objectives in reduced execution times for a broad set of testbed instances.
        <br/><br/>
        <u>Reference</u>:<br/>
        <b>A Multithreading Local Search For Multiobjective Energy-Aware Scheduling In Heterogeneous Computing Systems</b>, Iturriaga S., Nesmachnow S., Dorronsoro B., 26th European Conference on Modelling and Simulation (ECMS), Koblenz, Germany, 2012. 
</span>

<div> </div>


<%
// Below the application submission web form 
//
// The <form> tag contains a portlet parameter value called 'PortletStatus' the value of this item
// will be read by the processAction portlet method which then assigns a proper view mode before
// the call to the doView method.
// PortletStatus values can range accordingly to values defined into Enum type: Actions
// The processAction method will assign a view mode accordingly to the values defined into
// the Enum type: Views. This value will be assigned calling the function: setRenderParameter
//
%>
<span>
<form enctype="multipart/form-data" action="<portlet:actionURL portletMode="view"><portlet:param name="PortletStatus" value="ACTION_SUBMIT"/></portlet:actionURL>" method="post">
<dl>    
    <!-- This block contains: label, file input and textarea for GATE Macro file -->
    <dd>
        <!--
        Usage:
           bin/pals_cpu <scenario> <workload> <#tasks> <#machines> <algorithm> <#threads> <seed> <max time (secs)> <max iterations> <population size>

           Algorithms
               0 PALS 2-populations
               1 PALS 1-population
               2 MinMin
               3 MCT
               4 pMinMin
        -->
        <p>Scenario file <input type="file" name="file_scenario" id="file_scenarioId" accept="*.*" /></p>
        <p>Workload file <input type="file" name="file_workload" id="file_workloadId" accept="*.*" /></p>
        <p>Number of tasks <input type="text" name="ntasks" id="ntasksId" /></p>
        <p>Number of machines <input type="text" name="nmachines" id="nmachinesId" /></p>
        <p>Algorithm <select id="algorithmId" name="algorithm">
            <option value="1" selected>ME-MLS</option>
            <option value="2">MinMin</option>
            <option value="3">MCT</option>
            <option value="4">pMinMin</option>
        </select></p>
        <p>Number of threads <input type="text" name="nthreads" id="nthreadsId" /></p>
        <p>Random number generator seed <input type="text" name="randseed" id="randseedId" /></p>
        <p>Max. execution time <input type="text" name="timeout" id="timeoutId" /> seconds</p>
        <p>Max. iterations <input type="text" name="iterations" id="iterationsId" /></p>
        <p>Population size <input type="text" name="popsize" id="popsizeId" /></p>
    </dd>
    <!-- This block contains the experiment name -->
    <dd>
        <p>Insert below your <b>job identifyer</b></p>
        <textarea id="jobIdentifierId" rows="1" cols="60%" name="JobIdentifier">Job execution of: <%= date %></textarea>
    </dd>   
    <!-- This block contains form buttons: SUBMIT and Reset values -->
    <dd>
        <td><input type="button" value="Submit" onClick="preSubmit()"></td> 
    </dd>
</dl>
</form>
   <tr>
        <form action="<portlet:actionURL portletMode="HELP"> /></portlet:actionURL>" method="post">
        <td><input type="submit" value="About"></td>
        </form>        
   </tr>
</table>
</span>
</div>


<%
// Below the javascript functions used by the DPPM web form 
%>
<script language="javascript">
//
// preSubmit
//
function preSubmit() {  
    var ok = true;
    var missingFields = '';

    var scenario = document.getElementById('file_scenarioId');
    var workload = document.getElementById('file_workloadId');
    var ntasks = document.getElementById('ntasksId');
    var nmachines = document.getElementById('nmachinesId');
   
    if(scenario.value == '') {
        missingFields += '  Scenario file missing\n';
        ok = false;
    }
    if(workload.value == '') {
        missingFields += '  Workload file missing\n';
        ok = false;
    }
    if(ntasks.value == '') {
        missingFields += '  Number of tasks missing\n';
        ok = false;
    }
    if(nmachines.value == '') {
        missingFields += '  Number of machines missing\n';
        ok = false;
    }
    
    var nthreads = document.getElementById('nthreadsId');
    var randseed = document.getElementById('randseedId');
    var timeout = document.getElementById('timeoutId');
    var iterations = document.getElementById('iterationsId');
    var popsize = document.getElementById('popsizeId');

    if(nthreads.value == '') {
        missingFields += '  Number of threads missing\n';
        ok = false;
    }
    if(randseed.value == '') {
        missingFields += '  Random seed number missing\n';
        ok = false;
    }
    if(timeout.value == '') {
        missingFields += '  Max. execution time missing\n';
        ok = false;
    }
    if(iterations.value == '') {
        missingFields += '  Max. iterations missing\n';
        ok = false;
    }
    if(popsize.value == '') {
        missingFields += '  Population size missing\n';
        ok = false;
    }

    var jobId = document.getElementById('jobIdentifierId');
    if(jobId.value == '') {
        missingFields += '  Job identifier missing\n';
        ok = false;
    }    
    
    if(ok) 
    {
        document.forms[0].submit();
    }
    else 
    {
        alert("You cannot send an inconsisten job submission! \nMissing fields: \n" + missingFields);
    }
}

</script>
