// ## ================= Working code OLD ===============================
import React, { useState, useEffect } from 'react'
import {Row, Col, Form, Button, Badge} from 'react-bootstrap'
import HomeLayoutCss from './HomeLayout.module.css'
import { verified, pending, hand, summaryAI, DownloadIcon } from '../../assets/images';
import axios from 'axios';
import { messageService } from '../Utils/messageService'
import KycSummaryCss from './KycSummary.module.css'
import documentUploadCss from './DocumentUpload.module.css'
import LayoutLoading from './../Utils/LayoutLoading'
import UpdateAttributes from './UpdateAttributes';
import CustomModal from '../Utils/CustomModal';
import KycReport from './KycReport';
import downloadFile from '../Utils/downloadHelper';
import { selectEntityDetails } from '../../redux/entityDetails/selector';
import { selectSummaryDetails } from '../../redux/summaryDetails/selector';
import { connect } from 'react-redux';
import { Subject } from 'rxjs';
import { navigationService } from '../Utils/NavigationService';
import store from '../../redux/store';
import { setSummaryDetails } from '../../redux/summaryDetails/action';
import { HOME_CASE_CONFIG } from '../Utils/homeCaseConfig';

function Home({ entityDetails, summaryDetails, setSummaryDetails }) {

const caseType = entityDetails?.caseType || 'INDONESIA';
const caseConfig =
  HOME_CASE_CONFIG[caseType] || HOME_CASE_CONFIG.INDONESIA;

  const stageInfo = [
   {name: 'Documents', status: ' Pending', isActive: false},
   {name: 'Review', status: ' Pending', isActive: false},
  // {name: 'Screening', status: ' Completed', isActive: false},
   {name: 'Fulfilment', status: ' Pending', isActive: false},
   {name: 'Approval', status: ' Pending', isActive: true},
  ];

  const [approvalComment, setApprovalComment] = useState("");
  const [approvalDecision, setApprovalDecision] = useState("");

  const docs = [];
  const [stages, setStages] = useState(stageInfo);
  const [loading, setLoading] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [summary, setSummary] = useState('');
  const [riskRating, setRiskRating] = useState({risk_category: '', risk_score: ''})

  const clickHandler = (stageName, e) => {
    stages.forEach(stage => {
      stage.isActive = stage.name === stageName ? true : false
    })
    setStages([...stages]);
  }

  const reportHandler = () => {
    setShowReport(true);
  }

  const navigationHandler = (e) => {
    navigationService.sendMessage(e);
  }


  const DownloadReport = () => {
    // Support both entity (entityName) and individual (customerName) formats
    // Fallback to known individual name for development/testing
    const name = entityDetails?.customerName || entityDetails?.entityName || 'Aisyah Rahmani'
    
    // Ensure only Aisyah Rahmani can download reports
    if (name.toLowerCase() !== 'aisyah rahmani') {
      messageService.sendMessage({ variant: 'danger', message: 'Only Aisyah Rahmani is authorized to download reports' })
      return
    }
    
    setLoading(true)
    downloadFile({ endpoint: '/download_entity_final_report', body: { entity: name }, filename: `KYC_Report_${name}.pdf` })
      .then(() => {
        messageService.sendMessage({ variant: 'success', message: 'KYC report downloaded' })
      })
      .catch(err => {
        console.error('KYC download failed', err)
        messageService.sendMessage({ variant: 'danger', message: 'Report download failed: ' + (err?.message || 'server problem') })
      })
      .finally(() => {
        setLoading(false)
      })
  }

  useEffect(() => {
      setLoading(true)
      axios.get(`${process.env.REACT_APP_API_BASE_URL}/access_entity_risk`)
      .then(response => {
          setRiskRating(response.data)
          setLoading(false)
      })
      .catch(error => {
          messageService.sendMessage({variant:"danger", message:"server problem"})
      })
    
    
  }, [])

  // useEffect(() => {
  //   const fetchSummary = async () => {
  //     if (!summaryDetails || Object.keys(summaryDetails).length === 0) {
  //       setLoading(true);
  //       try {
  //         const response = await axios.get(`${process.env.REACT_APP_API_BASE_URL}/entity_kyc_summary`);
  //         if (typeof setSummaryDetails === 'function') {
  //           setSummaryDetails(response.data);
  //         } else {
  //           store.dispatch(setSummaryDetails(response.data));
  //         }
  //       } catch (error) {
  //         messageService.sendMessage({ variant: "danger", message: "server problem" });
  //       } finally {
  //         setLoading(false);
  //       }
  //     }
  //   };
  //   fetchSummary();
  // }, [summaryDetails, setSummaryDetails]);

  return (
    <div className={HomeLayoutCss.container}>
        <Row style={{backgroundColor: "var(--bodyColor)", border: "1px solid grey", height: "435px"}}>
            <Col md={4}>
              {
              stages.map(stage => {
                return (
                  <>
                    <div className={HomeLayoutCss.stage}>
                      <div className={HomeLayoutCss.stageName} onClick={() => clickHandler(stage.name)}>
                        <div style={{ display: "inline-block", paddingTop: "7px", width: "70px" }}>{stage.name}</div>
                      </div>
                      <div className={HomeLayoutCss.stageStatus}>
                        {stage.status === ' Completed' ?
                          <>
                            <img style={{ width: "25px" }} src={verified} alt="completed" />
                            <span>{stage.status}</span>
                          </> :
                          <>
                            <img style={{ width: "25px" }} src={pending} alt="pending" />
                            <span style={{ color: "red" }}>{stage.status}</span>
                          </>
                        }
                      </div>
                    </div>

                    <div className={HomeLayoutCss.stageDetails}>
                      {stage.name === "Documents" && stage.isActive &&
                        <>
                          <ul style={{ margin: "3px 0 0 5px" }}>
                            <li><a onClick={() => navigationHandler('missingDocuments')} className={HomeLayoutCss.navigationLink}>List of Missing Documents(if any)</a>:</li>
                            {/* <li>Passport, Utility Bill, Payslips</li> */}
                            <li>{caseConfig.RM_comments.documents.missingList.join(', ')}</li>
                          </ul>
                          <span style={{ marginLeft: "5px" }}>
                            <span style={{ fontFamily: "var(--poppinsSemiBold)" }}>Final Status: </span>
                            {/* Documents listed above are not available */}
                            {caseConfig.RM_comments.documents.finalStatus}
                          </span>
                        </>
                      }

                      {stage.name === "Review" && stage.isActive &&
                        <>
                          {/* <ul style={{ margin: "3px 0 0 5px" }}>
                            <li><a onClick={() => navigationHandler('missingAttributes')} className={HomeLayoutCss.navigationLink}>KYC Review Findings</a>:</li>
                            {caseConfig.review.findings}
                          </ul> */}
                          <span style={{ marginLeft: "5px" }}>
                            <span style={{ fontFamily: "var(--poppinsSemiBold)" }}>Final Status: </span>
                            {caseConfig.RM_comments.review.finalStatus}
                          </span>
                        </>
                      }

                      {stage.name === "Fulfilment" && stage.isActive &&
                        <>
                          <ul style={{ margin: "3px 0 0 5px" }}>
                            <p style={{ marginBottom: "0", fontFamily: "var(--poppinsSemiBold)" }}>List of updated KYC Attributes:</p>
                            {caseConfig.RM_comments.fulfilment.status}
                          </ul>
                          <span style={{ marginLeft: "5px" }}>
                            <span style={{ fontFamily: "var(--poppinsSemiBold)" }}>Final Status: </span>
                            {caseConfig.RM_comments.fulfilment.finalStatus}
                          </span>
                        </>
                      }

                      {stage.name === "Approval" && stage.isActive &&
                        <div>
                          <Form.Control
                            as="textarea"
                            name="inputText"
                            placeholder="Please Add Your Comment here..."
                            value={approvalComment}
                            onChange={(e) => setApprovalComment(e.target.value)}
                            style={{resize:"none", height:"50px", width: "360px", margin: "auto", border: "1px solid black", marginTop: "2px"}}
                            className={HomeLayoutCss.textArea}
                            required
                          />
                          <Form.Select
                            value={approvalDecision}
                            onChange={(e) => setApprovalDecision(e.target.value)}
                            style={{
                              width: "150px",
                              minHeight: "25px",
                              textAlign: "left",
                              fontFamily: "PoppinsRegular",
                              fontSize: "10px",
                              fontWeight: "500",
                              border: "1px solid black",
                              color: approvalDecision ? "#130202" : "#6c757d"
                            }}
                          >
                            <option value="" disabled>
                              Select Action
                            </option>
                            <option value="Approve">Approve</option>
                            <option value="EDD">EDD</option>
                            <option value="Reject">Reject</option>
                            <option value="Exit">Exit</option>
                          </Form.Select>
                          <Button
                            variant="success"
                            className={HomeLayoutCss.submit}
                            onClick={() => {
                              if (!approvalDecision) {
                                alert("Please select an action");
                                return;
                              }
                              console.log("Comment:", approvalComment);
                              console.log("Decision:", approvalDecision);
                              alert("Submitted successfully!");
                              setApprovalComment("");   // reset
                              setApprovalDecision("");  // reset
                            }}
                          >
                            Submit
                          </Button>
                        </div>
                      }
                    </div>
                  </>
                )
              })
            }


              {/* Commented for Individual KYC task as hand icon for submit button is not required */}
              {/* <img src={hand} alt="hand" /><span className={HomeLayoutCss.comment}>Reviewer's comment & Approval is required for submission of KYC Report for QA Check </span> */}
            </Col>
            <Col md={8}>
              <div className={HomeLayoutCss.summaryContainer}>
                <div style={{marginBottom:"2px"}}>
                  <img src={summaryAI} alt="summary" />
                  <span className={HomeLayoutCss.summaryTitle}>Case Summary</span>
                  <span className={HomeLayoutCss.report} onClick={reportHandler}>KYC Report</span>
                  <img src={DownloadIcon} alt="download" style={{position:"absolute", right:"115px", width:"30px", height:"25px", cursor: "pointer"}} onClick={DownloadReport}/>
                </div>
                <div className={HomeLayoutCss.summaryDiv}>
                <Row style={{backgroundColor: "#fff", width: "655px", marginLeft: "2px", height: "30px"}}>
                    <Col md={7}>
                    <span className={KycSummaryCss.riskRating}><span>Risk Category: </span>
                      {
                        riskRating.risk_category === "Low" ?
                       <Badge bg="success" className={documentUploadCss.badge}>{riskRating.risk_category}</Badge> :
                       <Badge bg="danger" className={documentUploadCss.badge}>{riskRating.risk_category}</Badge>
                      }
                    </span>
                    </Col>
                    <Col md={5}>
                    <span className={KycSummaryCss.riskRating}><span> Enhance Due Diligence Required</span>
                    <Badge bg="success" className={documentUploadCss.badge}>No</Badge></span>
                    </Col>
                </Row>
                <div style={{overflowY: "auto", height:"305px"}}>
                <div>
                {/* { 
                 summaryDetails &&
                    <ul>
                        <span className={KycSummaryCss.summaryHeader}>caseConfig</span>
                        <li>{summaryDetails.customer_profile_overview}</li>
                        <span className={KycSummaryCss.summaryHeader}>Overall Risk Assessment</span>
                        <li>{summaryDetails.overall_risk_assesment}</li>
                        <span className={KycSummaryCss.summaryHeader}>KYC Documentation</span>
                        <li>{summaryDetails.kyc_documentation}</li>
                        <span className={KycSummaryCss.summaryHeader}>Adverse Media Screening Risk</span>
                        <li>{summaryDetails.adverse_media_screening_risk}</li>
                        <span className={KycSummaryCss.summaryHeader}>PEP Status</span>
                        <li>{summaryDetails.pep_status}</li>
                        <span className={KycSummaryCss.summaryHeader}>Sanctions Screening</span>
                        <li>{summaryDetails.sanctions_screening}</li>
                        <span className={KycSummaryCss.summaryHeader}>SOW Due Diligence</span>
                        <li>{summaryDetails.sow_due_diligence}</li>
                      </ul>
                } */}
                {caseConfig?.caseSummary &&
                  <ul>
                    {Object.entries(caseConfig.caseSummary).map(([key, summaryItem]) => (
                      <React.Fragment key={key}>
                        <span className={KycSummaryCss.summaryHeader}>{summaryItem.label}</span>
                        <li>{summaryItem.value}</li>
                      </React.Fragment>
                    ))}
                  </ul>
                }
                </div>
                {/* <Row style={{backgroundColor: "#fff", width: "650px", margin: "2px", height: "30px"}}>
                <span className={KycSummaryCss.riskRating}><span style={{display:"inline-block",padding:"5px"}}>List of KYC Attributes to be updated </span></span>
                </Row> */}
                <UpdateAttributes />
                </div>
                </div>
              </div>
            </Col>
        </Row>
        {
                loading &&
                <LayoutLoading message="Generating Summary"/>
        }
        {
          showReport &&
          <CustomModal onHideHandler={() => setShowReport(false)} modalHeader='Report' size="lg">
            <KycReport />
          </CustomModal>
        }
    </div>
  )
}

const mapStateToProp = (state) => {
  return {
      entityDetails: selectEntityDetails(state),
      summaryDetails: selectSummaryDetails(state)
  }
}

const mapDispatchToProp = {
  setSummaryDetails
}

export default connect(mapStateToProp, mapDispatchToProp)(Home)
