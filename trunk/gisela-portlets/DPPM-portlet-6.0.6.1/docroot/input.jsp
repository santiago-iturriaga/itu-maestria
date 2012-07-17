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
<%//
  // DPPM 1.0.0 Submission Form
  //
  // The form has 2 input areas respectively for:
  //    * input instance file
  //    * Theta parameter value
  // Beside each text area a upolad button takes as input the 
  // file name related to one of the above fields.
  // The ohter buttons of the form are used for:
  //    o SUBMIT:        Used to execute the job on the eInfrastructure
  //    o Reset values:  Used to reset input fields
  //
  
  // Gets the current timestamp
  java.util.Date date = new java.util.Date();
%>

<%
// Below the descriptive area of the DPPM web form 
%>
<div class="MyPortletWebapp">
    <!-- portlet content here -->
<span>
	<img width="155" height="37" src="<%=renderRequest.getContextPath()%>/images/logo.jpg">
	
	<span style="position:relative;left:8px;top:0px">
	<p style="background:#5EA3CE;width:440px; height:50px;position:relative;left:140px;top:-35px;">
		<span style="font-size:18.0pt;font-family:Arial;color:white;position:relative;left:10px;top:15px;">DPPM</span>
		<span style="font-family:Arial;color:white;font-weight:bold;position:relative;left:20px;top:15px;">Deadline Problem in Project Management</span>
	</p>
  	</span>
</span>

<div> </div>
	<span style="position:relative;left:4px;top:-35px;width:581px;height:4px">
	<img width="581" height="4" src="<%=renderRequest.getContextPath()%>/images/bar.gif" v:shapes="_x0000_s1028">
	</span>

<div> </div>

<span style="position:relative; left:0px;top:-10px">	In general, project management involves planning and organizing a set of activities in order to generate a product or offer a service in the best possible way. A project duration can often be reduced by accelerating some of its activities by employing additional resources that increase the cost of the entire project. In this case, each activity can be performed by using a set of alternatives modes which are defined by a time-cost pair. Usually, only a reduced number of modes are taken into account for each activity. A key problem consists in finding a schedule that assigns modes to activities, providing a good
		tradeoff between the duration and cost of each activity, enabling the best project performance. 
		<br>
		This website describes the Deadline Problem in Project Management (DPPM), which accounts for both precedence between activities and deadline for its execution. In the related literature, it is also known as the Discrete Time/Cost Trade-off Problem (DTCTP).
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
 		<p><b>Application' input file</b> <input type="file" name="file_inputFile" id="upload_inputFileId" accept="*.*" onchange="uploadInputFile()"/></p>
		<p> Theta parameter
		<select id="thetaID" name="thetaParam">
			<option value="0.15">0.15</option>
			<option value="0.30" selected>0.30</option>
			<option value="0.45">0.45</option>
		</select>
		</p>
	</dd>
	<!-- This block contains the experiment name -->
	<dd>
		<p>Insert below your <b>job identifyer</b></p>
		<textarea id="jobIdentifierId" rows="1" cols="60%" name="JobIdentifier">Job execution of: <%= date %></textarea>
	</dd>	
	<!-- This block contains form buttons: SUBMIT and Reset values -->
  	<dd>
  		<td><input type="button" value="Submit" onClick="preSubmit()"></td> 
  		<td><input type="reset" value="Reset values" onClick="resetForm()"></td>
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
    var inputFileName=document.getElementById('upload_inputFileId');
    var inputFileText=document.getElementById('thetaID');
    var jobIdentifier=document.getElementById('jobIdentifierId');
    var state_inputFileName=false;
    var state_inputFileText=false;
    var state_jobIdentifier=false;
    
    if(inputFileName.value=="") state_inputFileName=true;
    //if(inputFileText.value=="" || inputFileText.value=="Insert here your text file, or upload a file") state_inputFileText=true;
    if(jobIdentifier.value=="") state_jobIdentifier=true;    
       
    var missingFields="";
    if(state_inputFileName) missingFields+="  Input file or Text message\n";
    if(state_jobIdentifier) missingFields+="  Job identifier\n";
    if(missingFields == "") {
      document.forms[0].submit();
    }
    else {
      alert("You cannot send an inconsisten job submission!\nMissing fields:\n"+missingFields);
        
    }
}

//
//  resetForm
//
// This function is responsible to enable all textareas
// when the user press the 'reset' form button
function resetForm() {
	var currentTime = new Date();
	var inputFileName=document.getElementById('upload_inputFileId');
	var inputFileText=document.getElementById('thetaID');
	var jobIdentifier=document.getElementById('jobIdentifierId');
        
        // Enable the textareas
	inputFileText.disabled=false;
	inputFileName.disabled=false;        			
            
	// Reset the job identifier
        jobIdentifier.value="Job execution of: "+currentTime.getDate()+"/"+currentTime.getMonth()+"/"+currentTime.getFullYear()+" - "+currentTime.getHours()+":"+currentTime.getMinutes()+":"+currentTime.getSeconds();
}
</script>
