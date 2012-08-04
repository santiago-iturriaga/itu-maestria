/**************************************************************************
Copyright (c) 2011:
Istituto Nazionale di Fisica Nucleare (INFN), Italy
Consorzio COMETA (COMETA), Italy

See http://www.infn.it and and http://www.consorzio-cometa.it for details on
the copyright holders.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author <a href="mailto:riccardo.bruno@ct.infn.it">Riccardo Bruno</a>(COMETA)
****************************************************************************/
package it.infn.ct;

// Import generic java libraries
import java.io.*;
import java.util.Iterator;
import java.util.List;
import java.util.Calendar;
import java.text.SimpleDateFormat;

// Importing portlet libraries
import javax.portlet.*;

// Importing liferay libraries
import com.liferay.portal.theme.ThemeDisplay;
import com.liferay.portal.kernel.util.WebKeys;
import com.liferay.portal.model.User;

// Importing Apache libraries
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.commons.fileupload.*;
import org.apache.commons.fileupload.disk.DiskFileItemFactory;
import org.apache.commons.fileupload.portlet.PortletFileUpload;

// Importing GridEngine Job libraries
import it.infn.ct.GridEngine.Job.*;
import it.infn.ct.GridEngine.Job.MultiInfrastructureJobSubmission;

//
// This is the class that overrides the GenericPortlet class methods
// You can create your own portlet just customizing the code skeleton
// available below. It provides mainly a working example on:
//    1) How to manage combination of Actions/Views
//    2) How to manage portlet preferences and help
//    3) How to show information using the Log object
//    4) How to execute a distributed application with GridEngine
//
public class ME_MLS_portlet extends GenericPortlet {

    // AppLogger class (No customizations needed)
    // Although developers can use System.out.println to watch their own console outputs
    // the use of Java logs is highly recommended.
    // Java Log object offers different output levels to show information:
    //    trace
    //    debug
    //    info
    //    warn
    //    error
    //    fatal
    // All of them accept a String as parameter containing the proper message to show.
    // AppLogger class uses  LogLevel eunerated type to express the log level verbosity
    // the setLogLevel method allows the portlet to print-out all logs types equal
    // or below the given log level accordingly to the priority:
    //       trace,debug,info,warn,erro,fatal
    private enum LogLevels {
        trace,
        debug,
        info,
        warn,
        error,
        fatal
    }
    // The AppLogger class wraps the apache.common Log object allowing the user to
    // enable/disable log accordingly to a given loglevel; the higher is the level
    // more verbose will be the produced output
    private class AppLogger {
        // Values associated
        private static final int   TRACE_LEVEL=6;
        private static final int   DEBUG_LEVEL=5;
        private static final int    INFO_LEVEL=4;
        private static final int    WARN_LEVEL=3;
        private static final int   ERROR_LEVEL=2;
        private static final int   FATAL_LEVEL=1;
        private static final int UNKNOWN_LEVEL=0;

        private Log _log;
        private int logLevel=AppLogger.INFO_LEVEL;

        public void setLogLevel(String level) {
            switch(LogLevels.valueOf(level)) {
                case trace:
                    logLevel=AppLogger.TRACE_LEVEL;
                    break;
                case debug:
                    logLevel=AppLogger.DEBUG_LEVEL;
                    break;
                case info:
                    logLevel=AppLogger.INFO_LEVEL;
                    break;
                case warn:
                    logLevel=AppLogger.WARN_LEVEL;
                    break;
                case error:
                    logLevel=AppLogger.ERROR_LEVEL;
                    break;
                case fatal:
                    logLevel=AppLogger.FATAL_LEVEL;
                    break;
                default:
                    logLevel=AppLogger.UNKNOWN_LEVEL;
            }
        }
        public AppLogger(Class cname) {
            _log = LogFactory.getLog(cname);
        }
        public void trace(String s) {
            if(   _log.isTraceEnabled()
               && logLevel >= AppLogger.TRACE_LEVEL)
                  _log.trace(s);
        }
        public void debug(String s) {
            if(   _log.isDebugEnabled()
               && logLevel >= AppLogger.DEBUG_LEVEL)
                  _log.trace(s);
        }
        public void info(String s) {
            if(   _log.isInfoEnabled()
               && logLevel >= AppLogger.INFO_LEVEL)
                  _log.info(s);
        }
        public void warn(String s) {
            if(   _log.isWarnEnabled()
               && logLevel >= AppLogger.WARN_LEVEL)
                  _log.warn(s);
        }
        public void error(String s) {
            if(   _log.isErrorEnabled()
               && logLevel >= AppLogger.ERROR_LEVEL)
                  _log.error(s);
        }
        public void fatal(String s) {
            if(   _log.isFatalEnabled()
               && logLevel >= AppLogger.FATAL_LEVEL)
                  _log.fatal(s);
        }
    } // AppLogger

    // Instantiate the logger object
    AppLogger _log = new AppLogger(ME_MLS_portlet.class);

    // This portlet uses Aciont/Views enumerations in order to
    // manage the different portlet modes and the corresponding
    // view to display
    // You may override the current values with your own business
    // logic best identifiers and manage them through: jsp and java code
    // The jsp parameter PortletStatus will be the responsible of
    // portlet mode switching. This parameter will be read by
    // the processAction method who will select the proper view mode
    // registering again into 'PortletStatus' renderResponse parameter
    // the next view mode.
    // The default prortlet mode by default is: ACTION_INPUT (see ProcessAction)
    private enum Actions {
         ACTION_INPUT    // Called before to show the INPUT view
        ,ACTION_SUBMIT   // Called after the user press the submit button
        ,ACTION_PILOT    // The user wants to modify the selected pilot script
        }

    private enum Views {
         VIEW_INPUT      // View containing application input fields
        ,VIEW_SUBMIT     // View reporting the job submission
        ,VIEW_PILOT      // Shows the pilot script and makes it editable
    }

    // Infrastructure information class stores the required information
    // to submit a job into a given e-Infrastructure.
    class Info_Infrastructure {
         String enableInfrastructure;
         String nameInfrastructure;
         String acronymInfrastructure;
         String bdiiHost;
         String wmsHosts;
         String pxServerHost;
         String pxServerPort;
         String pxServerSecure;
         String pxRobotId;
         String pxRobotVO;
         String pxRobotRole;
         String pxRobotRenewalFlag;
         String softwareTags;

         Info_Infrastructure() {
             enableInfrastructure
            =nameInfrastructure
            =acronymInfrastructure
            =bdiiHost
            =wmsHosts
            =pxServerHost
            =pxServerPort
            =pxServerSecure
            =pxRobotId
            =pxRobotVO
            =pxRobotRole
            =pxRobotRenewalFlag
            =softwareTags
            ="";
         }
         // Dump the content of infrastructure
         String dumpInfrastructure() {
         String dump=LS+"    enableInfrastructure : '"+enableInfrastructure +"'"
                    +LS+"    nameInfrastructure   : '"+nameInfrastructure   +"'"
                    +LS+"    acronymInfrastructure: '"+acronymInfrastructure+"'"
                    +LS+"    bdiiHost             : '"+bdiiHost             +"'"
                    +LS+"    wmsHosts             : '"+wmsHosts             +"'"
                    +LS+"    pxServerHost         : '"+pxServerHost         +"'"
                    +LS+"    pxServerPort         : '"+pxServerPort         +"'"
                    +LS+"    pxRobotId            : '"+pxRobotId            +"'"
                    +LS+"    pxRobotRole          : '"+pxRobotRole          +"'"
                    +LS+"    pxRobotVO            : '"+pxRobotVO            +"'"
                    +LS+"    softwareTags         : '"+softwareTags         +"'"
                    +LS;
         return dump;
         }
    }

    // The init values will be read form portlet.xml from <init-param> xml tag
    // This tag will be useful to setup defaults values for your own portlet
    class App_Init {
        String                portletVersion;
        String                logLevel;
        String                numInfrastructures;
        Info_Infrastructure[] infoInfra;
        String                pxUserProxy;
        String                sciGwyAppId;
        String                sciGwyUserTrackingDB_Hostname;
        String                sciGwyUserTrackingDB_Username;
        String                sciGwyUserTrackingDB_Password;
        String                sciGwyUserTrackingDB_Database;
        String                jobRequirements;
        String                pilotScript;

        public App_Init() {
            portletVersion
           =logLevel
           =numInfrastructures
           =pxUserProxy
           =sciGwyAppId
           =sciGwyUserTrackingDB_Hostname
           =sciGwyUserTrackingDB_Username
           =sciGwyUserTrackingDB_Password
           =sciGwyUserTrackingDB_Database
           =jobRequirements
           =pilotScript
           ="";
           // Initialize Info Infrastructure
           infoInfra=null;
        }
    } // App_Init

    // Instanciate the App_Init object
    App_Init appInit = new App_Init();

    // This object is used to store the values of portlet preferences
    // The init method will initialize their values with corresponding init_*
    // variables when the portlet first starts (see init_Preferences var).
    // Please notice that not all init_* variables have a corresponding pref_* value
    class App_Preferences {
        String                logLevel;
        String                numInfrastructures;
        Info_Infrastructure[] infoInfra;
        String                pxUserProxy;
        String                sciGwyAppId;
        String                jobRequirements;
        String                pilotScript;

        // This values ranges from 1 to numInfrastructures
        // It is used by the preference edit pane to scroll
        // among inserted infrastructures
        int                   paneInfrastucture;
        int                   inumInfrastructures;

        public App_Preferences() {
            logLevel
           =pxUserProxy
           =sciGwyAppId
           =jobRequirements
           =pilotScript
           ="";

           // Initialize Info Infrastructure
           infoInfra=null;

           // Initialize the paneInfrastructure;
           paneInfrastucture=0;
        } // App_Preferences

        // Set the number of infrastructures
        public void setNumInfrastructures(String numInfras) {
            numInfrastructures=numInfras;
            inumInfrastructures=Integer.parseInt(numInfras);
        } // setNumInfrastructures
        // Returns the number of infrastructures
        public int getNumInfrastructures() {
            return inumInfrastructures;
        } // getNumInfrastructure
        public int getCurrInfrastructure() {
            return paneInfrastucture;
        } // getCurrInfrastructure
        public void switchNextInfrastructure() {
            if(inumInfrastructures > 0) {
                paneInfrastucture++;
                if(paneInfrastucture > inumInfrastructures)
                    paneInfrastucture=1;
            }
        } // switchNextInfrastructure
        public void switchPreviousInfrastructure() {
            if(inumInfrastructures > 0) {
                paneInfrastucture--;
                if(paneInfrastucture <= 0)
                    paneInfrastucture=inumInfrastructures;
            }
        } // switchPreviousInfrastructure
        // Delete the current infrastructure shifting left the array
        public void delCurrInfrastructure() {
            // It is not possible to have zero Infrastructures
            if(inumInfrastructures > 1) {
                    // Shift left the Infrastructure preferences
                    for(int i=paneInfrastucture; i<= inumInfrastructures-1; i++) {
                        infoInfra[i-1].enableInfrastructure =infoInfra[i].enableInfrastructure;
                        infoInfra[i-1].nameInfrastructure   =infoInfra[i].nameInfrastructure;
                        infoInfra[i-1].acronymInfrastructure=infoInfra[i].acronymInfrastructure;
                        infoInfra[i-1].bdiiHost             =infoInfra[i].bdiiHost;
                        infoInfra[i-1].wmsHosts             =infoInfra[i].wmsHosts;
                        infoInfra[i-1].pxServerHost         =infoInfra[i].pxServerHost;
                        infoInfra[i-1].pxServerPort         =infoInfra[i].pxServerPort;
                        infoInfra[i-1].pxServerSecure       =infoInfra[i].pxServerSecure;
                        infoInfra[i-1].pxRobotId            =infoInfra[i].pxRobotId;
                        infoInfra[i-1].pxRobotVO            =infoInfra[i].pxRobotVO;
                        infoInfra[i-1].pxRobotRole          =infoInfra[i].pxRobotRole;
                        infoInfra[i-1].pxRobotRenewalFlag   =infoInfra[i].pxRobotRenewalFlag;
                        infoInfra[i-1].softwareTags         =infoInfra[i].softwareTags;
                    }
                    // The removed infrastructure is the last in the array
                    if(inumInfrastructures==paneInfrastucture)
                            paneInfrastucture--;
                    // Now reduce the number of insfrastructures
                    setNumInfrastructures(""+(--inumInfrastructures));
            }
        } // delCurrInfrastructure
        // Adds a new infrastructure
        public void addNewInfrastructure() {
            setNumInfrastructures(""+(++inumInfrastructures));
            // Create the new array
            Info_Infrastructure[] new_infoInfrastructure = new Info_Infrastructure[inumInfrastructures];
            // Copy all values from old the old array
            for(int i=0; i<inumInfrastructures-1; i++) {
                Info_Infrastructure new_infoInfra= new Info_Infrastructure();
                new_infoInfra.enableInfrastructure =infoInfra[i].enableInfrastructure;
                new_infoInfra.nameInfrastructure   =infoInfra[i].nameInfrastructure;
                new_infoInfra.acronymInfrastructure=infoInfra[i].acronymInfrastructure;
                new_infoInfra.bdiiHost             =infoInfra[i].bdiiHost;
                new_infoInfra.wmsHosts             =infoInfra[i].wmsHosts;
                new_infoInfra.pxServerHost         =infoInfra[i].pxServerHost;
                new_infoInfra.pxServerPort         =infoInfra[i].pxServerPort;
                new_infoInfra.pxServerSecure       =infoInfra[i].pxServerSecure;
                new_infoInfra.pxRobotId            =infoInfra[i].pxRobotId;
                new_infoInfra.pxRobotVO            =infoInfra[i].pxRobotVO;
                new_infoInfra.pxRobotRole          =infoInfra[i].pxRobotRole;
                new_infoInfra.pxRobotRenewalFlag   =infoInfra[i].pxRobotRenewalFlag;
                new_infoInfra.softwareTags         =infoInfra[i].softwareTags;
                new_infoInfrastructure[i]=new_infoInfra;
            } // Copy infrastructure settings
            // Last infrastructure remaining
            Info_Infrastructure new_infoInfra= new Info_Infrastructure();
            new_infoInfra.enableInfrastructure ="yes/no";
            new_infoInfra.nameInfrastructure   ="Infrastructure name";
            new_infoInfra.acronymInfrastructure="Infrastructure acronym";
            new_infoInfra.bdiiHost             ="BDII host";
            new_infoInfra.wmsHosts             ="WMS host";
            new_infoInfra.pxServerHost         ="Robot proxy server host";
            new_infoInfra.pxServerPort         ="Robot proxy server port";
            new_infoInfra.pxServerSecure       ="Robot proxy server secure flag";
            new_infoInfra.pxRobotId            ="Robot proxy id";
            new_infoInfra.pxRobotVO            ="Robot proxy VO";
            new_infoInfra.pxRobotRole          ="Robot proxy Role";
            new_infoInfra.pxRobotRenewalFlag   ="Robot proxy renewal flag";
            new_infoInfra.softwareTags         ="Software tags";
            new_infoInfrastructure[inumInfrastructures-1]=new_infoInfra;
            // Link the new array of infrastructures
            infoInfra=new_infoInfrastructure;
            // Point to the new infrastructure
            paneInfrastucture=inumInfrastructures;
        } // addNewInfrastructure

        // Makes a dump of a given infrastructure
        String dumpInfrastructure(int numInfrastructure) {
            String dump="";
            if(getNumInfrastructures() > 0) {
                dump =LS+"Infrastructure #"+numInfrastructure;
                dump+=LS+infoInfra[numInfrastructure-1].dumpInfrastructure();

            }
            return dump;
        } // dumpInfrastructure
        // Makes a dump of all infrastructures
        String dumpInfrastructures() {
            String dump="";
            for(int i=0; i<getNumInfrastructures(); i++) {
                int j=i+1;
                dump+=dumpInfrastructure(j);
            } // for all infrastructures
            return dump;
        } // dumpInfrastructures
        // Makes a dump of all preferences
        String dumpPreferences() {
            String dump=LS+"Preference values:"
                       +LS+"------------------"
                       +LS+"pref_logLevel          : '"+logLevel               +"'"
                       +LS+"pref_sciGwyAppId       : '"+sciGwyAppId            +"'"
                       +LS+"pref_numInfrastructures: '"+numInfrastructures     +"'"
                       +LS+"pref_currInfrastructure: '"+getCurrInfrastructure()+"'"
                       +LS+dumpInfrastructures()
                       +LS+"pref_jobRequirements   : '"+jobRequirements        +"'"
                       +LS+"pref_pilotScript       : '"+pilotScript            +"'"
                       +LS+"pref_pxUserProxy       : '"+pxUserProxy            +"'"
                       +LS;
            return dump;
        } // dumpPreferences
    } // App_Preferences

    // Instanciate the App_Preferences object
    App_Preferences appPreferences = new App_Preferences();

    //
    // Application input values
    //
    // Job submission values are collected inside a single object
    class App_Input {
        // Applicatoin inputs
        String scenarioFileName;   // Filename for application input file
        String workloadFileName;   // Filename for application input file
        String ntasks;
        String nmachines;
        String algorithm;
        String nthreads;
        String randseed;
        String timeout;
        String iterations;
        String popsize;
        String jobIdentifier;       // User' given job identifier

        // Each inputSandobox file must be declared below
        // This variable contains the content of an uploaded file
        String inputSandbox_scenarioInputFile;
        String inputSandbox_workloadInputFile;

        // Some user level information
        // must be stored as well
        String username;
        String timestamp;

        public App_Input() {
           scenarioFileName = workloadFileName = ntasks = nmachines = algorithm = nthreads = randseed = timeout = iterations = popsize = jobIdentifier
           = inputSandbox_scenarioInputFile = inputSandbox_workloadInputFile = username = timestamp = "";
        }
    } // App_Input

    // Liferay user data
    // Classes below are used by this portlet code to get information
    // about the current user
    public ThemeDisplay themeDisplay; // Liferay' ThemeDisplay variable
    public User user;                 // From ThemeDisplay get User data
    public String username;           // From User data the username

    // Liferay portlet data
    PortletSession portletSession; // PorteltSession
    PortletContext portletContext; // PortletContext
    public String  appServerPath;  // This variable stores the absolute path of the Web applications

    // Other misc valuse
    // (!) Pay attention that altough the use of the LS variable
    //     the replaceAll("\n","") has to be used
    public String LS = System.getProperty("line.separator");

    // Users must have separated inputSandbox files
    // these file will be generated into /tmp directory
    // and prefixed with the format <timestamp>_<user>_*
    // The timestamp format is:
    public static final String tsFormat = "yyyyMMddHHmmss";

    //
    // Portlet Methods
    //

    //
    // init
    //
    // The init method will be called when installing the portlet for the first time
    // This is the right time to get default values from WEBINF/portlet.xml file
    // Those values will be assigned into parameters the first time the processAction
    // will be called thanks to the appPreferences object
    //
    @Override
    public void init()
    throws PortletException
    {
        // Load default values from portlet.xml
        appInit.portletVersion                     = ""+getInitParameter("portletVersion");
        appInit.logLevel                           = ""+getInitParameter("logLevel");
        appInit.numInfrastructures                 = ""+getInitParameter("numInfrastructures");
        int numInfra=Integer.parseInt(appInit.numInfrastructures);
        _log.info("Number of infrastructures: '"+numInfra+"'");
        if(numInfra >0)
              appInit.infoInfra = new Info_Infrastructure[numInfra];
        for(int i=0; i<numInfra; i++) {
            int j=i+1;
            appInit.infoInfra[i] = new Info_Infrastructure();
            appInit.infoInfra[i].enableInfrastructure  = ""+getInitParameter(j+"_enableInfrastructure");
            appInit.infoInfra[i].nameInfrastructure    = ""+getInitParameter(j+"_nameInfrastructure");
            appInit.infoInfra[i].acronymInfrastructure = ""+getInitParameter(j+"_acronymInfrastructure");
            appInit.infoInfra[i].bdiiHost              = ""+getInitParameter(j+"_bdiiHost");
            appInit.infoInfra[i].wmsHosts              = ""+getInitParameter(j+"_wmsHosts");
            appInit.infoInfra[i].pxServerHost          = ""+getInitParameter(j+"_pxServerHost");
            appInit.infoInfra[i].pxServerPort          = ""+getInitParameter(j+"_pxServerPort");
            appInit.infoInfra[i].pxServerSecure        = ""+getInitParameter(j+"_pxServerSecure");
            appInit.infoInfra[i].pxRobotId             = ""+getInitParameter(j+"_pxRobotId");
            appInit.infoInfra[i].pxRobotVO             = ""+getInitParameter(j+"_pxRobotVO");
            appInit.infoInfra[i].pxRobotRole           = ""+getInitParameter(j+"_pxRobotRole");
            appInit.infoInfra[i].pxRobotRenewalFlag    = ""+getInitParameter(j+"_pxRobotRenewalFlag");
            appInit.infoInfra[i].softwareTags          = ""+getInitParameter(j+"_softwareTags");
        }
        appInit.pxUserProxy                        = ""+getInitParameter("pxUserProxy");
        appInit.sciGwyAppId                        = ""+getInitParameter("sciGwyAppId");
        appInit.sciGwyUserTrackingDB_Hostname      = ""+getInitParameter("sciGwyUserTrackingDB_Hostname");
        appInit.sciGwyUserTrackingDB_Username      = ""+getInitParameter("sciGwyUserTrackingDB_Username");
        appInit.sciGwyUserTrackingDB_Password      = ""+getInitParameter("sciGwyUserTrackingDB_Password");
        appInit.sciGwyUserTrackingDB_Database      = ""+getInitParameter("sciGwyUserTrackingDB_Database");
        appInit.jobRequirements                    = ""+getInitParameter("jobRequirements");
        // WARNING: Although the pilot script field is considered here it is not
        // Possible to specify a bash script code inside thie init_pilotScript
        // xml field. The content of pilot script must be inserted manually upon
        // the portlet installation through its configuration pane.
        appInit.pilotScript = ""+getInitParameter("pilotScript");
        appInit.pilotScript = appInit.pilotScript.replaceAll("\r", "");

        // Assigns the log level
        _log.setLogLevel(appInit.logLevel);

        // Get information about all infrastructures
        String infrastructuresInfrormations="";
        for(int i=0; i<numInfra; i++) {
            int j=i+1;
            infrastructuresInfrormations+=
               LS+"Infrastructure #"+j
              +LS+"  enableInfrastructure  : '"+appInit.infoInfra[i].enableInfrastructure  +"'"
              +LS+"  nameInfrastructures   : '"+appInit.infoInfra[i].nameInfrastructure    +"'"
              +LS+"  acronymInfrastructures: '"+appInit.infoInfra[i].acronymInfrastructure +"'"
              +LS+"  bdiiHost              : '"+appInit.infoInfra[i].bdiiHost              +"'"
              +LS+"  wmsHosts              : '"+appInit.infoInfra[i].wmsHosts              +"'"
              +LS+"  pxServerHost          : '"+appInit.infoInfra[i].pxServerHost          +"'"
              +LS+"  pxServerPort          : '"+appInit.infoInfra[i].pxServerPort          +"'"
              +LS+"  pxServerSecure        : '"+appInit.infoInfra[i].pxServerSecure        +"'"
              +LS+"  pxRobotId             : '"+appInit.infoInfra[i].pxRobotId             +"'"
              +LS+"  pxRobotVO             : '"+appInit.infoInfra[i].pxRobotVO             +"'"
              +LS+"  pxRobotRole           : '"+appInit.infoInfra[i].pxRobotRole           +"'"
              +LS+"  pxRobotRenewalFlag    : '"+appInit.infoInfra[i].pxRobotRenewalFlag    +"'"
              +LS+"  softwareTags          : '"+appInit.infoInfra[i].softwareTags          +"'"
              +LS;
        }
        // Show loaded values into log
        _log.info(
               LS+"Loading default values "
              +LS+"-----------------------"
              +LS+"portletVersion               : '"+appInit.portletVersion                     +"'"
              +LS+"logLevel                     : '"+appInit.logLevel                           +"'"
              +LS+"numInfrastructures           : '"+appInit.numInfrastructures                 +"'"
              +LS+infrastructuresInfrormations
              +LS+"pxUserProxy                  : '"+appInit.pxUserProxy                        +"'"
              +LS+"sciGwyAppId                  : '"+appInit.sciGwyAppId                        +"'"
              +LS+"sciGwyUserTrackingDB_Hostname: '"+appInit.sciGwyUserTrackingDB_Hostname      +"'"
              +LS+"sciGwyUserTrackingDB_Username: '"+appInit.sciGwyUserTrackingDB_Username      +"'"
              +LS+"sciGwyUserTrackingDB_Password: '"+appInit.sciGwyUserTrackingDB_Password      +"'"
              +LS+"sciGwyUserTrackingDB_Database: '"+appInit.sciGwyUserTrackingDB_Database      +"'"
              +LS+"jobRequirements              : '"+appInit.jobRequirements                    +"'"
              +LS+"pilotScript                  : '"+appInit.pilotScript                        +"'"
              +LS
              +LS+"!WARNING: Although the pilot script field is considered into the portlet.xml"
              +LS+"it is not possible to specify a bash script code inside the init_pilotScript"
              +LS+"xml' field. The content of the pilot script must be inserted manually upon"
              +LS+"the portlet installation through its configuration pane."
              +LS);
    } // init

    //
    // processAction
    //
    // This method allows the portlet to process an action request; this method is normally
    // called upon each user interaction (a submit button inside a jsp' <form statement)
    //
    @Override
    public void processAction(ActionRequest request, ActionResponse response)
        throws PortletException, IOException
    {
        _log.info("calling processAction ...");

        // Determine the username
        themeDisplay = (ThemeDisplay)request.getAttribute(WebKeys.THEME_DISPLAY);
        user         = themeDisplay.getUser();
        username     = user.getScreenName();
        _log.info("User: '"+user+"'");

        // Determine the application pathname
        portletSession = request.getPortletSession();
        portletContext = portletSession.getPortletContext();
        appServerPath  = portletContext.getRealPath("/");
        _log.info("Web Application path: '"+appServerPath+"'");

        // Determine the current portlet mode and forward this state to the response
        // Accordingly to JSRs168/286 the standard portlet modes are:
        // VIEW, EDIT, HELP
        PortletMode mode = request.getPortletMode();
        response.setPortletMode(request.getPortletMode());

        // Switch among different portlet modes: VIEW, EDIT, HELP
        // Custom modes are not covered by this template
        if (mode.equals(PortletMode.VIEW)){
            // The VIEW mode is the normal portlet mode where normal portlet
            // content will be shown to the user
            _log.info("Portlet mode: VIEW");

            // The actionStatus value will be taken from the calling jsp file
            // through the 'PortletStatus' parameter; the corresponding
            // VIEW mode will be stored registering the portlet status
            // as render parameter. See the call to setRenderParameter
            // If the actionStatus parameter is null or empty the default
            // action will be the ACTION_INPUT (input form)
            // This happens the first time the portlet is shown
            // The PortletStatus variable is managed by jsp and this java code
            String actionStatus=request.getParameter("PortletStatus");
            // Assigns the default ACTION
            if(   null==actionStatus
               || actionStatus.equals(""))
                actionStatus=""+Actions.ACTION_INPUT;

            // Different actions will be performed accordingly to the
            // different possible statuses
            switch(Actions.valueOf(actionStatus)) {
                case ACTION_INPUT:
                    _log.info("Got action: 'ACTION_INPUT'");

                    // Assign the correct view
                    response.setRenderParameter("PortletStatus",""+Views.VIEW_INPUT);
                break;
                case ACTION_PILOT:
                    _log.info("Got action: 'ACTION_PILOT'");
                    // Stores the new pilot script
                    String pilotScript=request.getParameter("pilotScript");
                    pilotScript.replaceAll("\r", "");
                    storeString(appServerPath+"WEB-INF/job/"+appPreferences.pilotScript,pilotScript);
                    // Assign the correct view
                    response.setPortletMode(PortletMode.EDIT);
                break;
                case ACTION_SUBMIT:
                    _log.info("Got action: 'ACTION_SUBMIT'");

                    // Get current preference values
                    getPreferences(request,null);

                    // Create the appInput object
                    App_Input appInput = new App_Input();

                    // Stores the user submitting the job
                    appInput.username=username;

                    // Determine the submissionTimeStamp
                    SimpleDateFormat dateFormat = new SimpleDateFormat(tsFormat);
                    String timestamp = dateFormat.format(Calendar.getInstance().getTime());
                    appInput.timestamp=timestamp;

                    // Process input fields and files to upload
                    getInputForm(request,appInput);

                    // Submit the job
                    submitJob(appInput);

                    // Send the jobIdentifier and assign the correct view
                    response.setRenderParameter("PortletStatus",""+Views.VIEW_SUBMIT);
                    response.setRenderParameter("jobIdentifier",""+appInput.jobIdentifier);
                break;
                default:
                     _log.info("Unhandled action: '"+actionStatus+"'");
                     response.setRenderParameter("PortletStatus",""+Views.VIEW_INPUT);
            }
        }
        else if(mode.equals(PortletMode.HELP)) {
            // The HELP mode used to give portlet usage HELP to the user
            // This code will be called after the call to doHelp method
            _log.info("Portlet mode: HELP");
        }
        else if(mode.equals(PortletMode.EDIT)) {
            // The EDIT mode is used to view/setup portlet preferences
            // This code will be called after the user sends the actionURL
            // generated by the doEdit method
            // The code below just stores new preference values or
            // reacts to the preference settings changes
            _log.info("Portlet mode: EDIT");

            // New application preferences will takem from edit.jsp
            String newpref_logLevel    = ""+request.getParameter("pref_logLevel");
            String newpref_sciGwyAppId = ""+request.getParameter("pref_sciGwyAppId");

            // Retrieve the current ifnrstructure in preference
            int numInfrastructures=appPreferences.getNumInfrastructures();
            int currInfra=appPreferences.getCurrInfrastructure();

            _log.info(
                   LS+"Number of infrastructures: '"+numInfrastructures  +"'"
                  +LS+"currentInfrastructure:     '"+currInfra           +"'"
                  +LS);

            // Take care of the preference action (Infrastructure preferences)
            // <,>,+,- buttons
            String pref_action=""+request.getParameter("pref_action");
            _log.info("pref_action: '"+pref_action+"'");

            // Reacts to the Infrastructure change and
            // Determine the next view mode (return to the input pane)
            if(pref_action.equalsIgnoreCase("next")) {
                appPreferences.switchNextInfrastructure();
                _log.info("Got next infrastructure action; switching to: '"+appPreferences.paneInfrastucture+"'");
            }
            else if(pref_action.equalsIgnoreCase("previous")) {
                appPreferences.switchPreviousInfrastructure();
                _log.info("Got prev infrastructure action; switching to: '"+appPreferences.paneInfrastucture+"'");
            }
            else if(pref_action.equalsIgnoreCase("add")) {
                appPreferences.addNewInfrastructure();
                storePreferences(request);
                _log.info("Got add infrastructure action; current infrastrucure is now: '"+appPreferences.paneInfrastucture+"'");
                return;
            }
            else if(pref_action.equalsIgnoreCase("remove")) {
                appPreferences.delCurrInfrastructure();
                storePreferences(request);
                _log.info("Got remove infrastructure action; current infrastrucure is now: '"+appPreferences.paneInfrastucture+"' and infrastructures: '"+appPreferences.numInfrastructures+"'");
                return;
            }
            else if(pref_action.equalsIgnoreCase("done")) {
                // None of the above actions selected; return to the VIEW mode
                response.setPortletMode(PortletMode.VIEW);
                response.setRenderParameter("PortletStatus", ""+Views.VIEW_INPUT);
                return;
            }
            else if(pref_action.equalsIgnoreCase("viewPilot")) {
                int i=currInfra-1;
                // None of the above actions selected; return to the VIEW mode
                response.setPortletMode(PortletMode.VIEW);
                response.setRenderParameter("PortletStatus",""+Views.VIEW_PILOT);
                response.setRenderParameter("pilotScript"
                                            ,updateString(appServerPath+"WEB-INF/job/"+appPreferences.pilotScript));
                return;
            }
            else {
                // No other special actions to do ...
            }

            // Take the values of the current infrastructure
            // Current infrastructure pane held by appPreferences.paneInfrastucture value
            Info_Infrastructure newpref_infraInfo = new Info_Infrastructure();
            if(appPreferences.getCurrInfrastructure()>0) {
                newpref_infraInfo.enableInfrastructure = ""+request.getParameter("pref_enableInfrastructure");
                newpref_infraInfo.nameInfrastructure   = ""+request.getParameter("pref_nameInfrastructure");
                newpref_infraInfo.acronymInfrastructure= ""+request.getParameter("pref_acronymInfrastructure");
                newpref_infraInfo.bdiiHost             = ""+request.getParameter("pref_bdiiHost");
                newpref_infraInfo.wmsHosts             = ""+request.getParameter("pref_wmsHosts");
                newpref_infraInfo.pxServerHost         = ""+request.getParameter("pref_pxServerHost");
                newpref_infraInfo.pxServerPort         = ""+request.getParameter("pref_pxServerPort");
                newpref_infraInfo.pxServerSecure       = ""+request.getParameter("pref_pxServerSecure");
                newpref_infraInfo.pxRobotId            = ""+request.getParameter("pref_pxRobotId");
                newpref_infraInfo.pxRobotVO            = ""+request.getParameter("pref_pxRobotVO");
                newpref_infraInfo.pxRobotRole          = ""+request.getParameter("pref_pxRobotRole");
                newpref_infraInfo.pxRobotRenewalFlag   = ""+request.getParameter("pref_pxRobotRenewalFlag");
                newpref_infraInfo.softwareTags         = ""+request.getParameter("pref_softwareTags");
            } // Setup current infrastructure

            // Other application preferences
            String newpref_jobRequirements    = ""+request.getParameter("pref_jobRequirements");
            String newpref_pilotScript        = ""+request.getParameter("pref_pilotScript");
            String newpref_pxUserProxy        = ""+request.getParameter("pref_pxUserProxy");

            // Cleanup the pilot script from dangerous \r characters
            newpref_pilotScript=newpref_pilotScript.replaceAll("\r", "");

            // Show preference values changes
            String infrastructuresInformations="";
            if(currInfra > 0) {
                // Current preference values
                String    pref_enableInfrastructure =appPreferences.infoInfra[currInfra-1].enableInfrastructure;
                String    pref_nameInfrastructure   =appPreferences.infoInfra[currInfra-1].nameInfrastructure;
                String    pref_acronymInfrastructure=appPreferences.infoInfra[currInfra-1].acronymInfrastructure;
                String    pref_bdiiHost             =appPreferences.infoInfra[currInfra-1].bdiiHost;
                String    pref_wmsHosts             =appPreferences.infoInfra[currInfra-1].wmsHosts;
                String    pref_pxServerHost         =appPreferences.infoInfra[currInfra-1].pxServerHost;
                String    pref_pxServerPort         =appPreferences.infoInfra[currInfra-1].pxServerPort;
                String    pref_pxServerSecure       =appPreferences.infoInfra[currInfra-1].pxServerSecure;
                String    pref_pxRobotId            =appPreferences.infoInfra[currInfra-1].pxRobotId;
                String    pref_pxRobotVO            =appPreferences.infoInfra[currInfra-1].pxRobotVO;
                String    pref_pxRobotRole          =appPreferences.infoInfra[currInfra-1].pxRobotRole;
                String    pref_pxRobotRenewalFlag   =appPreferences.infoInfra[currInfra-1].pxRobotRenewalFlag;
                String    pref_softwareTags         =appPreferences.infoInfra[currInfra-1].softwareTags;
                // New preference values
                String newpref_enableInfrastructure =newpref_infraInfo.enableInfrastructure;
                String newpref_nameInfrastructure   =newpref_infraInfo.nameInfrastructure;
                String newpref_acronymInfrastructure=newpref_infraInfo.acronymInfrastructure;
                String newpref_bdiiHost             =newpref_infraInfo.bdiiHost;
                String newpref_wmsHosts             =newpref_infraInfo.wmsHosts;
                String newpref_pxServerHost         =newpref_infraInfo.pxServerHost;
                String newpref_pxServerPort         =newpref_infraInfo.pxServerPort;
                String newpref_pxServerSecure       =newpref_infraInfo.pxServerSecure;
                String newpref_pxRobotId            =newpref_infraInfo.pxRobotId;
                String newpref_pxRobotVO            =newpref_infraInfo.pxRobotVO;
                String newpref_pxRobotRole          =newpref_infraInfo.pxRobotRole;
                String newpref_pxRobotRenewalFlag   =newpref_infraInfo.pxRobotRenewalFlag;
                String newpref_softwareTags         =newpref_infraInfo.softwareTags;

                // Prepare the Log string with changes
                infrastructuresInformations+=
                     LS+"Infrastructure #"+appPreferences.paneInfrastucture
                    +LS+"  enableInfrastructure       : '"+pref_enableInfrastructure  +"' -> '"+newpref_enableInfrastructure +"'"
                    +LS+"  nameInfrastructures        : '"+pref_nameInfrastructure    +"' -> '"+newpref_nameInfrastructure   +"'"
                    +LS+"  acronymInfrastructures     : '"+pref_acronymInfrastructure +"' -> '"+newpref_acronymInfrastructure+"'"
                    +LS+"  bdiiHost                   : '"+pref_bdiiHost              +"' -> '"+newpref_bdiiHost             +"'"
                    +LS+"  wmsHosts                   : '"+pref_wmsHosts              +"' -> '"+newpref_wmsHosts             +"'"
                    +LS+"  pxServerHost               : '"+pref_pxServerHost          +"' -> '"+newpref_pxServerHost         +"'"
                    +LS+"  pxServerPort               : '"+pref_pxServerPort          +"' -> '"+newpref_pxServerPort         +"'"
                    +LS+"  pxServerSecure             : '"+pref_pxServerSecure        +"' -> '"+newpref_pxServerSecure       +"'"
                    +LS+"  pxRobotId                  : '"+pref_pxRobotId             +"' -> '"+newpref_pxRobotId            +"'"
                    +LS+"  pxRobotVO                  : '"+pref_pxRobotVO             +"' -> '"+newpref_pxRobotVO            +"'"
                    +LS+"  pxRobotRole                : '"+pref_pxRobotRole           +"' -> '"+newpref_pxRobotRole          +"'"
                    +LS+"  pxRobotRenewalFlag         : '"+pref_pxRobotRenewalFlag    +"' -> '"+newpref_pxRobotRenewalFlag   +"'"
                    +LS+"  softwareTags               : '"+pref_softwareTags          +"' -> '"+newpref_softwareTags         +"'"
                    +LS;
                // Assigns new values
                appPreferences.infoInfra[currInfra-1].enableInfrastructure =newpref_enableInfrastructure;
                appPreferences.infoInfra[currInfra-1].nameInfrastructure   =newpref_nameInfrastructure;
                appPreferences.infoInfra[currInfra-1].acronymInfrastructure=newpref_acronymInfrastructure;
                appPreferences.infoInfra[currInfra-1].bdiiHost             =newpref_bdiiHost;
                appPreferences.infoInfra[currInfra-1].wmsHosts             =newpref_wmsHosts;
                appPreferences.infoInfra[currInfra-1].pxServerHost         =newpref_pxServerHost;
                appPreferences.infoInfra[currInfra-1].pxServerPort         =newpref_pxServerPort;
                appPreferences.infoInfra[currInfra-1].pxServerSecure       =newpref_pxServerSecure;
                appPreferences.infoInfra[currInfra-1].pxRobotId            =newpref_pxRobotId;
                appPreferences.infoInfra[currInfra-1].pxRobotVO            =newpref_pxRobotVO;
                appPreferences.infoInfra[currInfra-1].pxRobotRole          =newpref_pxRobotRole;
                appPreferences.infoInfra[currInfra-1].pxRobotRenewalFlag   =newpref_pxRobotRenewalFlag;
                appPreferences.infoInfra[currInfra-1].softwareTags         =newpref_softwareTags;
            } // for each Infrastructure
            _log.info(
                 LS+"variable name          : 'Old Value' -> 'New value'"
                +LS+"---------------------------------------------------"
                +LS+"pref_logLevel          : '"+appPreferences.logLevel          +"' -> '"+newpref_logLevel       +"'"
                +LS+"pref_SciGwyAppId       : '"+appPreferences.sciGwyAppId       +"' -> '"+newpref_sciGwyAppId    +"'"
                +LS+"pref_numInfrastructures: '"+appPreferences.numInfrastructures+"' -> '"+numInfrastructures     +"'"
                +LS+infrastructuresInformations
                +LS+"pref_jobRequirements   : '"+appPreferences.jobRequirements   +"' -> '"+newpref_jobRequirements+"'"
                +LS+"pref_pilotScript       : '"+appPreferences.pilotScript       +"' -> '"+newpref_pilotScript    +"'"
                +LS+"pref_pxUserProxy       : '"+appPreferences.pxUserProxy       +"' -> '"+newpref_pxUserProxy    +"'"
                +LS);

            // Assign the new variable to the preference object
            appPreferences.logLevel          =   newpref_logLevel;
            appPreferences.sciGwyAppId       =   newpref_sciGwyAppId;
            appPreferences.pxUserProxy       =   newpref_pxUserProxy;
            appPreferences.jobRequirements   =   newpref_jobRequirements;
            appPreferences.pilotScript       =   newpref_pilotScript;
            appPreferences.pxUserProxy       =   newpref_pxUserProxy;

            // Store new preferences
            storePreferences(request);
        }
        else {
            // Unsupported portlet modes come here
            _log.warn("Custom portlet mode: '"+mode.toString()+"'");
        }
    } // processAction

    //
    // Store preferences
    //
    // Uses the appPreference object settings to store Application parameters
    //
    void storePreferences(ActionRequest request)
    throws PortletException, IOException{
        // The code below stores all the portlet preference values
        PortletPreferences prefs = request.getPreferences();
        int nunInfrastructures=appPreferences.getNumInfrastructures();
        if(prefs!=null) {
            prefs.setValue("pref_logLevel"          , appPreferences.logLevel);
            prefs.setValue("pref_sciGwyAppId"       , appPreferences.sciGwyAppId);
            prefs.setValue("pref_numInfrastructures", appPreferences.numInfrastructures);
            for(int i=0; i<nunInfrastructures; i++) {
                int j=i+1;
                prefs.setValue("pref_"+j+"_enableInfrastructure" ,appPreferences.infoInfra[i].enableInfrastructure);
                prefs.setValue("pref_"+j+"_nameInfrastructure"   ,appPreferences.infoInfra[i].nameInfrastructure);
                prefs.setValue("pref_"+j+"_acronymInfrastructure",appPreferences.infoInfra[i].acronymInfrastructure);
                prefs.setValue("pref_"+j+"_bdiiHost"             ,appPreferences.infoInfra[i].bdiiHost);
                prefs.setValue("pref_"+j+"_wmsHosts"             ,appPreferences.infoInfra[i].wmsHosts);
                prefs.setValue("pref_"+j+"_pxServerHost"         ,appPreferences.infoInfra[i].pxServerHost);
                prefs.setValue("pref_"+j+"_pxServerPort"         ,appPreferences.infoInfra[i].pxServerPort);
                prefs.setValue("pref_"+j+"_pxServerSecure"       ,appPreferences.infoInfra[i].pxServerSecure);
                prefs.setValue("pref_"+j+"_pxRobotId"            ,appPreferences.infoInfra[i].pxRobotId);
                prefs.setValue("pref_"+j+"_pxRobotVO"            ,appPreferences.infoInfra[i].pxRobotVO);
                prefs.setValue("pref_"+j+"_pxRobotRole"          ,appPreferences.infoInfra[i].pxRobotRole);
                prefs.setValue("pref_"+j+"_pxRobotRenewalFlag"   ,appPreferences.infoInfra[i].pxRobotRenewalFlag);
                prefs.setValue("pref_"+j+"_softwareTags"         ,appPreferences.infoInfra[i].softwareTags);
            }
            prefs.setValue("pref_jobRequirements"   , appPreferences.jobRequirements);
            prefs.setValue("pref_pilotScript"       , appPreferences.pilotScript);
            prefs.setValue("pref_pxUserProxy"       , appPreferences.pxUserProxy);
            prefs.store();
        } // pref !=null
    } // storePreferences

    //
    // Method responsible to show portlet content to the user accordingly to the current view mode
    //
    @Override
    protected void doView(RenderRequest request, RenderResponse response)
    throws PortletException, IOException
    {
        _log.info("calling doView ...");
        response.setContentType("text/html");

        // Determine the application pathname
        portletSession = request.getPortletSession();
        portletContext = portletSession.getPortletContext();
        appServerPath  = portletContext.getRealPath("/");
        _log.info("Web Application path: '"+appServerPath+"'");

        // Switch among supported views; the currentView is determined by the
        // portlet render parameter value stored into PortletStatus identifier
        // this value has been assigned by the actionStatus or it will be
        // null in case the doView method will be called without a
        // previous processAction call; in such a case the default VIEW_INPIUT
        // will be selected.
        //The PortletStatus variable is managed by jsp and this java code
        String currentView=request.getParameter("PortletStatus");
        if(  null==currentView
           ||currentView.equals(""))
            currentView=""+Views.VIEW_INPUT;

        // Different actions will be performed accordingly to the
        // different possible view modes
        switch(Views.valueOf(currentView)) {
            // The following code is responsible to call the proper jsp file
            // that will provide the correct portlet interface
            case VIEW_INPUT: {
                _log.info("VIEW_INPUT Selected ...");
                PortletRequestDispatcher dispatcher=getPortletContext().getRequestDispatcher("/input.jsp");
                dispatcher.include(request, response);
            }
            break;
            case VIEW_PILOT: {
                _log.info("VIEW_PILOT Selected ...");
                String pilotScript = request.getParameter("pilotScript");
                request.setAttribute("pilotScript", pilotScript);
                PortletRequestDispatcher dispatcher=getPortletContext().getRequestDispatcher("/viewPilot.jsp");
                dispatcher.include(request, response);
            }
            break;
            case VIEW_SUBMIT: {
                _log.info("VIEW_SUBMIT Selected ...");
                String jobIdentifier = request.getParameter("jobIdentifier");
                request.setAttribute("jobIdentifier", jobIdentifier);
                PortletRequestDispatcher dispatcher=getPortletContext().getRequestDispatcher("/submit.jsp");
                dispatcher.include(request, response);
            }
            break;
            default:
                _log.info("Unknown view mode: "+currentView.toString());
        } // switch
    } // doView

    //
    // doEdit
    //
    // This methods prepares an actionURL that will be used by edit.jsp file into a <input ...> form
    // As soon the user press the action button the processAction will be called and the portlet mode
    // will be set as EDIT.
    @Override
    public void doEdit(RenderRequest request,RenderResponse response)
    throws PortletException,IOException {
        response.setContentType("text/html");

        // Get current preference values
        getPreferences(null,request);

        // Get the current infrastructure and the number of infrastructure
        int currInfra=appPreferences.getCurrInfrastructure();
        int numInfrastructures=appPreferences.getNumInfrastructures();

        // ActionURL and the current preference value will be passed to the edit.jsp
        PortletURL pref_actionURL = response.createActionURL();
        request.setAttribute("pref_actionURL",pref_actionURL.toString());
        // Send preference values
        request.setAttribute("pref_logLevel"          ,   appPreferences.logLevel);
        request.setAttribute("pref_sciGwyAppId"       ,   appPreferences.sciGwyAppId);
        request.setAttribute("pref_numInfrastructures",   appPreferences.numInfrastructures);
        request.setAttribute("pref_currInfrastructure",""+appPreferences.paneInfrastucture);
        // Send Infrastructure specific data
        if(   currInfra > 0
           && currInfra <= numInfrastructures) {
          request.setAttribute("pref_enableInfrastructure" ,appPreferences.infoInfra[currInfra-1].enableInfrastructure);
          request.setAttribute("pref_nameInfrastructure"   ,appPreferences.infoInfra[currInfra-1].nameInfrastructure);
          request.setAttribute("pref_acronymInfrastructure",appPreferences.infoInfra[currInfra-1].acronymInfrastructure);
          request.setAttribute("pref_bdiiHost"             ,appPreferences.infoInfra[currInfra-1].bdiiHost);
          request.setAttribute("pref_wmsHosts"             ,appPreferences.infoInfra[currInfra-1].wmsHosts);
          request.setAttribute("pref_pxServerHost"         ,appPreferences.infoInfra[currInfra-1].pxServerHost);
          request.setAttribute("pref_pxServerPort"         ,appPreferences.infoInfra[currInfra-1].pxServerPort);
          request.setAttribute("pref_pxServerSecure"       ,appPreferences.infoInfra[currInfra-1].pxServerSecure);
          request.setAttribute("pref_pxRobotId"            ,appPreferences.infoInfra[currInfra-1].pxRobotId);
          request.setAttribute("pref_pxRobotVO"            ,appPreferences.infoInfra[currInfra-1].pxRobotVO);
          request.setAttribute("pref_pxRobotRole"          ,appPreferences.infoInfra[currInfra-1].pxRobotRole);
          request.setAttribute("pref_pxRobotRenewalFlag"   ,appPreferences.infoInfra[currInfra-1].pxRobotRenewalFlag);
          request.setAttribute("pref_softwareTags"         ,appPreferences.infoInfra[currInfra-1].softwareTags);
        } // if paneInfrastructure > 0
        request.setAttribute("pref_jobRequirements"   ,appPreferences.jobRequirements);
        request.setAttribute("pref_pilotScript"       ,appPreferences.pilotScript);
        request.setAttribute("pref_pxUserProxy"       ,appPreferences.pxUserProxy);

        // The edit.jsp will be the responsible to show/edit the current preference values
        PortletRequestDispatcher dispatcher=getPortletContext().getRequestDispatcher("/edit.jsp");
        dispatcher.include(request, response);
    } // doEdit

    //
    // doHelp
    //
    // This method just calls the jsp responsible to show the portlet information
    @Override
    public void doHelp(RenderRequest request, RenderResponse response)
    throws PortletException,IOException {
        response.setContentType("text/html");
        request.setAttribute("portletVersion",appInit.portletVersion);
        PortletRequestDispatcher dispatcher=getPortletContext().getRequestDispatcher("/help.jsp");
        dispatcher.include(request, response);
    } // doHelp

    //
    // updateString
    //
    // This method takes as input a filename and will transfer its
    // content inside a String variable
    private String updateString(String file) throws IOException {
        String line;
        StringBuilder stringBuilder = new StringBuilder();
        BufferedReader reader = new BufferedReader( new FileReader (file));
        while((line = reader.readLine()) != null ) {
            stringBuilder.append(line);
            stringBuilder.append(LS);
        }
        return stringBuilder.toString();
    }

    //
    // storeString
    //
    // This method will transfer the content of a given String into
    // a given filename
    private void storeString(String fileName,String fileContent) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        writer.write(fileContent);
        writer.close();
    }

    //
    // getInputForm
    //
    // The use of upload file controls needs the use of "multipart/form-data"
    // form type. With this kind of input form it is necessary to process
    // each item of the action request manually
    //
    // All form' input items are identified by the 'name' input property
    // inside the jsp file
    private enum inputControlsIds {
        file_scenario,
        file_workload,
        ntasks,
        nmachines,
        algorithm,
        nthreads,
        randseed,
        timeout,
        iterations,
        popsize,
        JobIdentifier     // User defined Job identifier
    };
    //
    // getInputForm (method)
    //
    void getInputForm(ActionRequest request,App_Input appInput) {
        if (PortletFileUpload.isMultipartContent(request)) {
            try {
                String logstring="";
                
                File repositoryPath = new File("/tmp");
                
                DiskFileItemFactory diskFileItemFactory = new DiskFileItemFactory();
                diskFileItemFactory.setRepository(repositoryPath);

                FileItemFactory factory = new DiskFileItemFactory();
                PortletFileUpload upload = new PortletFileUpload(factory);
                List items = upload.parseRequest(request);            
                Iterator iter = items.iterator();
                
                while (iter.hasNext()) {
                    FileItem item = (FileItem)iter.next();
                    String   fieldName  =item.getFieldName();
                    String   fileName   =item.getName();
                    String   contentType=item.getContentType();
                    boolean  isInMemory =item.isInMemory();
                    long     sizeInBytes=item.getSize();
                    
                    // Prepare a log string with field list
                    logstring+=LS+"field name: '"+fieldName+"' - '"+item.getString()+"'";
                    
					String[] result;
					
                    switch(inputControlsIds.valueOf(fieldName)) {
                        case file_scenario:
							result = processInputFile(item, appInput);
						
                            appInput.scenarioFileName = result[1]; //item.getString();
                            appInput.inputSandbox_scenarioInputFile = result[0];
							
							logstring+=" ('"+appInput.inputSandbox_scenarioInputFile+"')";
                        break;
                        case file_workload:
							result = processInputFile(item, appInput);
						
                            appInput.workloadFileName = result[1]; //item.getString();
                            appInput.inputSandbox_workloadInputFile = result[0];
							
							logstring+=" ('"+appInput.inputSandbox_workloadInputFile+"')";
                        break;
                        case algorithm:
                            appInput.algorithm = item.getString();
                        break;
                        case nmachines:
                            appInput.nmachines = item.getString();
                        break;
                        case ntasks:
                            appInput.ntasks = item.getString();
                        break;
                        case nthreads:
                            appInput.nthreads = item.getString();
                        break;
                        case popsize:
                            appInput.popsize = item.getString();
                        break;
                        case randseed:
                            appInput.randseed = item.getString();
                        break;
                        case timeout:
                            appInput.timeout = item.getString();
                        break;
                        case iterations:
                            appInput.iterations = item.getString();
                        break;
                        case JobIdentifier:
                            appInput.jobIdentifier = item.getString();
                        break;
                        default:
                            _log.warn("Unhandled input field: '"+fieldName+"' - '"+item.getString()+"'");
							
                    } // switch fieldName
                } // while iter.hasNext()
                _log.info(
                       LS+"Reporting"
                      +LS+"---------"
                      +LS+logstring
                      +LS);
            } // try
            catch (Exception e) {
                _log.info("Caught exception while processing files to upload: '"+e.toString()+"'");
            }
        } else  {
			// The input form do not use the "multipart/form-data"
            // Retrieve from the input form the given application values
            appInput.scenarioFileName = "ouch!"; //(String)request.getParameter("file_scenario");
            appInput.workloadFileName = "ouch!!"; //(String)request.getParameter("file_workload");
            appInput.algorithm = (String)request.getParameter("algorithm");
            appInput.nmachines = (String)request.getParameter("nmachines");
            appInput.ntasks = (String)request.getParameter("ntasks");
            appInput.nthreads = (String)request.getParameter("nthreads");
            appInput.popsize = (String)request.getParameter("popsize");
            appInput.randseed = (String)request.getParameter("randseed");
            appInput.timeout = (String)request.getParameter("timeout");
            appInput.iterations = (String)request.getParameter("iterations");
            appInput.jobIdentifier=(String)request.getParameter("JobIdentifier");
        } // ! isMultipartContent

        // Show into the log the taken inputs
        _log.info(
               LS+"Taken input parameters:"
              +LS+"-----------------------"
              
              +LS+"scenarioFileName: '"+appInput.scenarioFileName+"'"
              +LS+"workloadFileName: '"+appInput.workloadFileName+"'"
              +LS+"algorithm: '"+appInput.algorithm+"'"
              +LS+"nmachines: '"+appInput.nmachines+"'"
              +LS+"ntasks: '"+appInput.ntasks+"'"
              +LS+"nthreads: '"+appInput.nthreads+"'"
              +LS+"popsize: '"+appInput.popsize+"'"
              +LS+"randseed: '"+appInput.randseed+"'"
              +LS+"timeout: '"+appInput.timeout+"'"
              +LS+"iterations: '"+appInput.iterations+"'"
              +LS+"jobIdentifier: '"+appInput.jobIdentifier+"'"
              +LS);
    } // getInputForm

    //
    // processInputFile
    //
    // This method is called when the user specifies a input file to upload
    // the file will be saved first into /tmp directory and then its content
    // stored into the corresponding String variable
    // Before to submit the job the String value will be stored in the
    // proper job inputSandbox file
    String[] processInputFile(FileItem item, App_Input appInput) {
        String theNewFileName = "";
		String uploadPath = "/tmp/";
        
        // Determin the filename
        String fileName = item.getName();
        if(!fileName.equals("")) {
            // Determine the fieldName
            String fieldName = item.getFieldName();

            // Create a filename for the uploaded file
            theNewFileName = appInput.timestamp + "_" + appInput.username + "_" + fileName;
            File uploadedFile = new File(uploadPath + theNewFileName);
            
            _log.info("Uploading file: '" + fileName + "' into '" + uploadPath + theNewFileName + "'");
            try {
                item.write(uploadedFile);
            }
            catch (Exception e) {
                _log.error("Caught exception while uploading file: 'file_inputFile'");
            }
        } // if
        
		String[] result = {uploadPath + theNewFileName, theNewFileName};
        return result;
    } // processInputFile

    //
    // getPreferences
    //
    // This method retrieves current portlet preference values and it can
    // be called by both processAction or doView methods
    private void getPreferences( ActionRequest actionRequest
                                ,RenderRequest renderRequest) {
        _log.info("Calling: getPreferences ...");
        PortletPreferences prefs=null;
        if(null!=actionRequest)
            prefs = actionRequest.getPreferences();
        else if(null != renderRequest)
            prefs = renderRequest.getPreferences();
        else _log.warn("Both render request and action request are null");

        if (null != prefs) {
            appPreferences.logLevel          =""+prefs.getValue("pref_logLevel"          ,appInit.logLevel);
            appPreferences.sciGwyAppId       =""+prefs.getValue("pref_sciGwyAppId"       ,appInit.sciGwyAppId);
            appPreferences.setNumInfrastructures(""+prefs.getValue("pref_numInfrastructures",appInit.numInfrastructures));
            // Now retrieves the infrastructures information
            int numInfras=appPreferences.getNumInfrastructures();

            // Allocate the array only if it is not initialized yet
            if(      0  < numInfras
               && null == appPreferences.infoInfra) {
                appPreferences.infoInfra = new Info_Infrastructure[numInfras];
                appPreferences.paneInfrastucture=1; // Initialize the edit pane number
            }
            // For each infrastructure
            // The preference name is indexed with the infrastructure number: 1,2,...
            String infrastructuresInfrormations="";
            for(int i=0; i<numInfras; i++) {
                int j=i+1;
                // Allocate the Info_Infrastructure class onlt if it is not initialized yet
                if(null == appPreferences.infoInfra[i]) {
                    appPreferences.infoInfra[i] = new Info_Infrastructure();
                }
                appPreferences.infoInfra[i].enableInfrastructure =""+prefs.getValue("pref_"+j+"_enableInfrastructure" ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].enableInfrastructure :"yes/no"));
                appPreferences.infoInfra[i].nameInfrastructure   =""+prefs.getValue("pref_"+j+"_nameInfrastructure"   ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].nameInfrastructure   :"Name infrastructure"));
                appPreferences.infoInfra[i].acronymInfrastructure=""+prefs.getValue("pref_"+j+"_acronymInfrastructure",(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].acronymInfrastructure:"Acronym infrastructure"));
                appPreferences.infoInfra[i].bdiiHost             =""+prefs.getValue("pref_"+j+"_bdiiHost"             ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].bdiiHost             :"BDII host"));
                appPreferences.infoInfra[i].wmsHosts             =""+prefs.getValue("pref_"+j+"_wmsHosts"             ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].wmsHosts             :"WMS host"));
                appPreferences.infoInfra[i].pxServerHost         =""+prefs.getValue("pref_"+j+"_pxServerHost"         ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].pxServerHost         :"Proxy server host"));
                appPreferences.infoInfra[i].pxServerPort         =""+prefs.getValue("pref_"+j+"_pxServerPort"         ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].pxServerPort         :"Proxy server port"));
                appPreferences.infoInfra[i].pxServerSecure       =""+prefs.getValue("pref_"+j+"_pxServerSecure"       ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].pxServerSecure       :"Proxy server secure connection flag"));
                appPreferences.infoInfra[i].pxRobotId            =""+prefs.getValue("pref_"+j+"_pxRobotId"            ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].pxRobotId            :"Proxy server robot identifier"));
                appPreferences.infoInfra[i].pxRobotVO            =""+prefs.getValue("pref_"+j+"_pxRobotVO"            ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].pxRobotVO            :"Proxy server robot VO"));
                appPreferences.infoInfra[i].pxRobotRole          =""+prefs.getValue("pref_"+j+"_pxRobotRole"          ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].pxRobotRole          :"Proxy server robot role"));
                appPreferences.infoInfra[i].pxRobotRenewalFlag   =""+prefs.getValue("pref_"+j+"_pxRobotRenewalFlag"   ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].pxRobotRenewalFlag   :"Proxy server renewal"));
                appPreferences.infoInfra[i].softwareTags         =""+prefs.getValue("pref_"+j+"_softwareTags"         ,(i<Integer.parseInt(appInit.numInfrastructures)?appInit.infoInfra[i].softwareTags         :"Software tags"));
            } // for each Infrastructure
            appPreferences.jobRequirements   =""+prefs.getValue("pref_jobRequirements"   ,appInit.jobRequirements);
            appPreferences.pilotScript       =""+prefs.getValue("pref_pilotScript"       ,appInit.pilotScript);
            appPreferences.pxUserProxy       =""+prefs.getValue("pref_pxUserProxy"       ,appInit.pxUserProxy);

            // Assigns the log level
            _log.setLogLevel(appPreferences.logLevel);

            // Show preference values into log
            _log.info(appPreferences.dumpPreferences());
        } // if
    } // getPreferences

    //
    // submitJob
    //
    // This method sends the job into the distributed infrastructure using
    // the GridEngine methods.
    void submitJob(App_Input appInput) {
        // Get the number of configured Infrastructures
        int numInfras=appPreferences.getNumInfrastructures();
        if(numInfras >0) {
            //
            // Initialize the GridEngine Multi Infrastructure Job Submission object
            //
            MultiInfrastructureJobSubmission miJobSubmission = new MultiInfrastructureJobSubmission();
            //
            // Determine the number of infrastructures
            //
            int numEnabledInfrastructures=0;
            for(int i=0; i<numInfras; i++)
              if(appPreferences.infoInfra[i].enableInfrastructure.equalsIgnoreCase("yes"))
                numEnabledInfrastructures++;
            _log.info("Enabled infrastructures: '"+numEnabledInfrastructures+"'");

            // Initialize the array of GridEngine' infrastructure objects
            InfrastructureInfo infrastructures[] = new InfrastructureInfo[numEnabledInfrastructures];
            // For each infrastructure
            for(int i=0,h=0; i<numInfras; i++) {
                int j=i+1;
                // Take care of wms list
                // GridEngine supports a list of WMSes that can be specified by the preference
                // value 'wmsHosts' separating each wms by the ';' character
                String wmsHostList[]=null;
                if(    null != appPreferences.infoInfra[i].wmsHosts
                   && !appPreferences.infoInfra[i].wmsHosts.equals("")) {
                    wmsHostList = appPreferences.infoInfra[i].wmsHosts.split(";");
                    String showWMSList=LS+"wmsHostList"
                                      +LS+"-----------"
                                      +LS;
                    for(int k=0; k<wmsHostList.length; k++)
                        showWMSList+=LS+wmsHostList[k];
                    _log.info(showWMSList);
                } // if wmsList
                if(appPreferences.infoInfra[i].enableInfrastructure.equalsIgnoreCase("yes")) {
                  // Build the infrastructure object and assign it to the infrastructure array
                  // (!)Not yet used values:
                  //    pxServerSecure
                  //    pxRobotRenewalFlag
                  infrastructures[h] = new InfrastructureInfo( appPreferences.infoInfra[i].acronymInfrastructure
                                                              ,appPreferences.infoInfra[i].bdiiHost
                                                              ,wmsHostList
                                                              ,appPreferences.infoInfra[i].pxServerHost
                                                              ,appPreferences.infoInfra[i].pxServerPort
                                                              ,appPreferences.infoInfra[i].pxRobotId
                                                              ,appPreferences.infoInfra[i].pxRobotRole
                                                              ,appPreferences.infoInfra[i].pxRobotVO
                                                              ,appPreferences.infoInfra[i].softwareTags);
                  // Add the infrastructure into the miJobSubmission object
                  miJobSubmission.addInfrastructure(infrastructures[h]);
                  h++;
                  // Shows the added infrastructure
                  _log.info(LS+appPreferences.dumpInfrastructure(j));
                } // Add enabled infrastructure
                else {
                  _log.info(LS+"Disabled infrastructure: "
                           +LS+appPreferences.dumpInfrastructure(j));

                }
            } // for each infrastructure

            //
            // Configure the job properties
            //
            // (!) Job level unused preference values
            //     pxUserProxy
            //     pxRobotRenewalFlag
            //

            // Application Id
            int applicationId=Integer.parseInt(appPreferences.sciGwyAppId);

            // UserTrakingDatabase
            String   hostUTDB=appInit.sciGwyUserTrackingDB_Hostname; // Grid Engine' UserTraking database host
            // Data below should not used by portlets
            //String  unameUTDB=appInit.sciGwyUserTrackingDB_Username; // Username
            //String passwdUTDB=appInit.sciGwyUserTrackingDB_Password; // Password
            //String dbnameUTDB=appInit.sciGwyUserTrackingDB_Database; // Database

            // Job details
            String executable="/bin/sh";                  // Application executable
            // bin/pals_cpu <scenario> <workload> <#tasks> <#machines> <algorithm> <#threads> <seed> <max time (secs)> <max iterations> <population size>
            String arguments =appPreferences.pilotScript
                +" "+appInput.scenarioFileName
                +" "+appInput.workloadFileName
                +" "+appInput.ntasks
                +" "+appInput.nmachines
                +" "+appInput.algorithm
                +" "+appInput.nthreads
                +" "+appInput.randseed
                +" "+appInput.timeout
                +" "+appInput.iterations
                +" "+appInput.popsize;
                // executable' arguments
            String outputPath="/tmp/";                    // Output Path
            String outputFile="ME-MLS-Output.txt";      // Distributed application standard output
            String errorFile ="ME-MLS-Error.txt";       // Distrubuted application standard error
            String appFile   ="ME-MLS-Files.tar.gz";    // Hostname output files (created by the pilot script)

            // InputSandbox (string with comma separated list of file names)
            String inputSandbox= appServerPath+"WEB-INF/job/"+appPreferences.pilotScript // pilot script
                               +","+appInput.inputSandbox_scenarioInputFile
                               +","+appInput.inputSandbox_workloadInputFile
                               +","+appServerPath+"WEB-INF/job/makefile"
                               +","+appServerPath+"WEB-INF/job/src.tar.gz"
                               ;
            // OutputSandbox (string with comma separated list of file names)
            String outputSandbox=appFile;                                     // Output file

            // Take care of job requirements
            // More requirements can be specified in the preference value 'jobRequirements'
            // separating each requirement by the ';' character
            String jdlRequirements[] = appPreferences.jobRequirements.split(";");
            int numRequirements=0;
            for(int i=0; i<jdlRequirements.length; i++) {
                if(!jdlRequirements[i].equals("")) {
                  jdlRequirements[numRequirements] = "JDLRequirements=("+jdlRequirements[i]+")";
                  numRequirements++;
                }
                _log.info("Requirement["+i+"]='"+jdlRequirements[i]+"'");
            } // for each jobRequirement

            // Other job initialization settings
            miJobSubmission.setExecutable(executable);     // Specify the executeable
            miJobSubmission.setArguments(arguments);       // Specify the application' arguments
            miJobSubmission.setOutputPath(outputPath);     // Specify the output directory
            miJobSubmission.setOutputFiles(outputSandbox); // Setup output files (OutputSandbox)
            miJobSubmission.setJobOutput(outputFile);      // Specify the std-outputr file
            miJobSubmission.setJobError(errorFile);        // Specify the std-error file
            if(   null != inputSandbox                     // Setup input files (InputSandbox) avoiding empty inputSandboxes
               && inputSandbox.length() > 0)
                miJobSubmission.setInputFiles(inputSandbox);

            // Submit Job
            miJobSubmission.submitJobAsync(username, hostUTDB, applicationId, appInput.jobIdentifier);

            // Show log
            // View jobSubmission details in the log
            _log.info(
               LS+"JobSent"
              +LS+"-------"
              +LS+"Executable    : '"+executable   +"'"
              +LS+"Arguments     : '"+arguments    +"'"
              +LS+"Output path   : '"+outputPath   +"'"
              +LS+"Output sandbox: '"+outputSandbox+"'"
              +LS+"Ouput file    : '"+outputFile   +"'"
              +LS+"Error file    : '"+errorFile    +"'"
              +LS+"Input sandbox : '"+inputSandbox +"'"
              +LS); // _log.info

        } // numInfra > 0
        else {
            _log.warn(
                   LS+"There are no infrastructures configured; impossible to send any job"
                  +LS+"Configure the application preferences in order to setup at least"
                  +LS+"an infrastructure."
                  +LS);
        } // numInfra == 0
   } // submitJob
} // DPPM_portlet
