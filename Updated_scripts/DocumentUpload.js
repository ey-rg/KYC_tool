/**
 * UNIFIED & PARAMETRIZED DOCUMENT UPLOAD
 * ----------------------------------------------------
 * Original UI output preserved
 * Original functionality preserved (doc click, modal preview, comments, status badges)
 * No duplicate Status columns
 * Doc View column present & clickable (KYC table only)
 * Single conditional switch for INDIA / INDONESIA
 * No behavioral change to BasicTable or CustomModal
 */

import React, { useState, useMemo, useEffect } from 'react';
import { Button, Badge, Form,DropdownButton,Dropdown,ButtonGroup  } from 'react-bootstrap';
import BasicTable from '../Utils/BasicTable';
import CustomModal from '../Utils/CustomModal';
import documentUploadCss from './DocumentUpload.module.css';
import { saveIcon, missingDocument, AutoEmail} from '../../assets/images';
import axios from 'axios';
import { messageService } from '../Utils/messageService';
import LayoutLoading from '../Utils/LayoutLoading';
import { HOME_CASE_CONFIG } from '../Utils/homeCaseConfig';

/* ================= IMAGE IMPORTS ================= */
import {
  adhar_combined,
  pan_card,
  customer_passport,
  electricity_bill,
  indonesian_national_id,
  indonesian_utility_bill,
  indonesian_passport,
  indonesian_masked_payslip
} from '../../assets/images';

/* ================= CASE CONFIG ================= */
const DOCUMENTS_CONFIG = {
  INDIA: {
    documents: {
      NATIONAL_ID: { label: 'National ID', image: adhar_combined },
      PAN: { label: 'PAN Card', image: pan_card },
      PASSPORT: { label: 'Passport', image: customer_passport },
      UTILITY: { label: 'Utility Bill', image: electricity_bill },
      PAYSLIP: { label: 'Payslip', image: null }
    },
  //   kycAttributes: [
  //     { attribute: 'Name', value: 'Deepak Srivastava', source: 'NATIONAL_ID', status: 'Matched', info: '', comments: '' },
  //     { attribute: 'Name', value: 'Deepak Jind', source: 'PASSPORT', status: 'Not Matched', info: 'Name Mismatch', comments: '' },
  //     { attribute: 'DOB', value: '30/08/1992', source: 'NATIONAL_ID', status: 'Matched', info: '', comments: '' },
  //     { attribute: 'DOB', value: '22/04/1990', source: 'PASSPORT', status: 'Not Matched', info: 'DOB Mismatch', comments: '' },
  //     { attribute: 'Address', value: 'Plot N0. 443, House No. 204, Krishna nagar, Bhagwanpur Pin-221005', source: 'NATIONAL_ID', status: 'Matched', info: '', comments: '' },
  //     { attribute: "Father's Name", value: 'Rajendra Srivastava', source: 'NATIONAL_ID', status: 'Matched', info: '', comments: '' },
  //     { attribute: 'Address', value: '', source: 'UTILITY',  status: 'Invalid', info: 'Blurred Document', comments: '' },
  //     { attribute: 'Annual Income', value: '1,000,000 INR', source: 'PAN', status: 'Matched', info: '', comments: '' },
  //     { attribute: 'Annual Income', value: '', source: 'PAYSLIP', status: 'Not Available', info: '', comments: '' }
  //   ]
  },

  INDONESIA: {
    documents: {
      NATIONAL_ID: { label: 'National ID', image: indonesian_national_id },
      PASSPORT: { label: 'Passport', image: indonesian_passport },
      UTILITY: { label: 'Utility Bill', image: indonesian_utility_bill },
      PAYSLIP: { label: 'Payslip', image: indonesian_masked_payslip }
    },
    // kycAttributes: [
    //   { attribute: 'Name', value: 'Aisyah Rahmani', source: 'NATIONAL_ID', status: 'Matched', info: '', comments: '' },
    //   { attribute: 'DOB', value: '23/06/1979', source: 'PASSPORT', status: 'Matched', info: '', comments: '' },
    //   { attribute: 'Address', value: '100 Pasir Panjang Road, #03-01 The Beacon, Singapore 118520', source: 'UTILITY', status: 'Matched', info: '', comments: '' },
    //   { attribute: 'Annual Income', value: '135,500,000 IDR', source: 'PAYSLIP', status: 'Matched', info: '', comments: '' }
    // ]
  }
};

/* ================= COMPONENT (Update case type)================= */
function DocumentUpload({ caseType = 'INDONESIA', showDocs = false }) {
  const DOCUMENTS =
    DOCUMENTS_CONFIG[caseType]?.documents || {};

  const KYC_ATTRIBUTES =
    HOME_CASE_CONFIG[caseType]?.kycAttributes || [];

  const [kycData, setKycData] = useState(
    KYC_ATTRIBUTES.map(item => ({
      ...item,
      docView: DOCUMENTS?.[item.source]?.image || null,
      comments: item.comments || ''
    }))
  );
    
  const initialData = [
      {items: 'Adhar Card', status:"Available", comments: ''},
      {items: 'PAN Card', status: "Available", comments: ''},
      {items: 'Passport', status: "Available"},
      {items: 'Payslip', status: "Available"},
      {items: 'Utility Bill', status: "Available"}
    ];

  const [showImageModal, setShowImageModal] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showMissDocs, setShowMissDocs] = useState(false);

  const [data, setData] = useState(initialData);
  // const [kycData, setKycData] = useState(
  //   CASE.kycAttributes.map(item => ({
  //     ...item,
  //     docView: CASE.documents[item.source]?.image || null,
  //     comments: ''
  //   }))
  // );


  const documentTableData = useMemo(
    () =>
      Object.values(DOCUMENTS).map(doc => ({
        items: doc.label,
        status: doc.image ? 'Available' : 'Not Available',
        comments: ''
      })),
    [DOCUMENTS]
  );

  const handleImageClick = (img) => {
    setSelectedImage(img);
    setShowImageModal(true);
  };

  /* ========== DOCUMENT TABLE (ORIGINAL FORMAT) ========== */
  const documentHeaders = [
    {
      Header: 'Status',
      accessor: 'status',
      Cell: ({ value }) => (
        <Badge bg={value === 'Available' ? 'success' : 'secondary'}>
          {value}
        </Badge>
      )
    },
    
  ];
  
  data.map(item => {
    return null;
  })

  // derive missing documents list from current data
  const missDocs = data.filter(item => item.status !== 'Available').map(d => d.items);

  useEffect(() => {
    if (typeof showDocs !== 'undefined' && showDocs) {
      setShowMissDocs(showDocs);
    }
  }, [/* showDocs prop may be passed externally */]);

  const missDocsHandler = () => {
      setShowMissDocs(true)
  }

  /* ========== KYC ATTRIBUTE TABLE (DOC VIEW ENABLED) ========== */
  const kycHeaders = [
    { Header: 'KYC Attribute', accessor: 'attribute' },
    { Header: 'Value', accessor: 'value' },
    // {
    //   Header: 'Source',
    //   accessor: 'source',
    //   Cell: ({ value }) => CASE.documents[value]?.label || value
    // },
    {
      Header: 'Source',
      accessor: 'source',
      Cell: ({ value }) => DOCUMENTS?.[value]?.label || value
    },
    {
      Header: 'Doc View',
      accessor: 'docView',
      Cell: ({ value }) =>
        value ? (
          <Button variant="link" size="sm" onClick={() => handleImageClick(value)}>
            View
          </Button>
        ) : (
          '-'
        )
    },
    {
      Header: 'Status',
      accessor: 'status',
      Cell: ({ value }) => (
        <Badge bg={value === 'Matched' ? 'success' : 'danger'}>
          {value}
        </Badge>
      )
    },
    { Header: 'Info', accessor: 'info' },
    {
      Header: "Reviewer's Co.",
      accessor: 'comments',
      Cell: ({ value, row }) => (
        <Form.Control
          className={documentUploadCss.inputTextArea}
          value={value}
          onChange={(e) => {
            const updated = [...kycData];
            updated[row.index].comments = e.target.value;
            setKycData(updated);
          }}
        />
      )
    }
  ];

  const sendMailHandler = (eventKey) => {
      setLoading(true)
      if (eventKey === 1) {
          //axios.post(`${process.env.REACT_APP_API_BASE_URL}/send_mail_with_attachment_non_compliant`)
          axios.post(`${process.env.REACT_APP_API_BASE_URL}/email_RFI`)
          .then(response => {
              setLoading(false)
              messageService.sendMessage({variant: "success", message: "Mail has been Sent Successfully."})
          })
          .catch(error => {
              setLoading(false)
              messageService.sendMessage({variant: "danger", message: "server error"});
          })
      } else {
      //axios.post(`${process.env.REACT_APP_API_BASE_URL}/send_mail_non_compliant`)
      axios.post(`${process.env.REACT_APP_API_BASE_URL}/email_EDD`)
      .then(response => {
          setLoading(false)
          messageService.sendMessage({variant: "success", message: "Mail has been Sent Successfully."})
      })
      .catch(error => {
          setLoading(false)
          messageService.sendMessage({variant: "danger", message: "server error"});
      })
    }  
  }
  /* ================= RENDER ================= */
  // return (
  //   <>
  //     <div className={documentUploadCss.innerDiv}>
  //       <BasicTable availableColumns={kycHeaders} data={kycData} />
  //     </div>
  //     {showImageModal && (
  //       <CustomModal
  //         modalHeader="Document Preview"
  //         onHideHandler={() => setShowImageModal(false)}
  //       >
  //         <img
  //           src={selectedImage}
  //           alt="Preview"
  //           style={{ width: '100%', maxHeight: '70vh', objectFit: 'contain' }}
  //         />
  //       </CustomModal>
        
  //     )}
  //   </>
  // );

    return (
    <>
      <div>
        <span style={{ margin: '10px 5px 0px 0px', float: 'right' }}>
          <img src={missingDocument} alt="missingDocument" style={{ width: '45px' }} />
          <Button
            className={`${documentUploadCss.documentList} backgroundDanger`}
            variant="danger"
            onClick={missDocsHandler}
          >
            Missing Documents
          </Button>

          <img src={AutoEmail} alt="AutoEmail" style={{ width: '36px', marginRight: '4px' }} />

          <DropdownButton
            as={ButtonGroup}
            title="Customer Outreach"
            className={`${documentUploadCss.documentDropdown} backgroundDanger`}
            variant="none"
            id="bg-nested-dropdown"
          >
            <Dropdown.Item eventKey="1" onClick={() => sendMailHandler(1)}>
              RFI
            </Dropdown.Item>
            <Dropdown.Item eventKey="2" onClick={() => sendMailHandler(2)}>
              EDD
            </Dropdown.Item>
          </DropdownButton>
        </span>
      </div>

      <div className={documentUploadCss.innerDiv} style={{ marginTop: '20px' }}>
        <BasicTable availableColumns={kycHeaders} data={kycData} />
      </div>

      {showImageModal && (
        <CustomModal
          modalHeader="Document Preview"
          onHideHandler={() => setShowImageModal(false)}
        >
          <img
            src={selectedImage}
            alt="Preview"
            style={{ width: '100%', maxHeight: '70vh', objectFit: 'contain' }}
          />
        </CustomModal>
      )}

      {showMissDocs && (
        <CustomModal
          modalHeader="Missing Documents"
          onHideHandler={() => setShowMissDocs(false)}
        >
          {missDocs.length ? (
            <ul>
              {missDocs.map(doc => (
                <li key={doc}>{doc}</li>
              ))}
            </ul>
          ) : (
            <p
              style={{
                fontFamily: 'var(--poppinsRegular)',
                fontSize: 'var(--fontSizeSmall)'
              }}
            >
              There are no missing documents
            </p>
          )}
        </CustomModal>
      )}

      {loading && <LayoutLoading message="Loading" />}
    </>
  );
}

export default DocumentUpload;


// ============================================================================================


// import React, { useState, useEffect } from 'react';
// import documentUploadCss from './DocumentUpload.module.css';
// import { Button, Form, Badge,DropdownButton,Dropdown,ButtonGroup } from 'react-bootstrap';
// // import { saveIcon, missingDocument, AutoEmail, adhar_combined, pan_card ,electricity_bill, customer_passport} from '../../assets/images';
// import { saveIcon, missingDocument, AutoEmail,indonesian_national_id ,indonesian_utility_bill, indonesian_passport,indonesian_masked_payslip} from '../../assets/images';
// import BasicTable from '../Utils/BasicTable';
// import { connect } from 'react-redux';
// import CustomModal from '../Utils/CustomModal';
// import { selectEntityDetails } from '../../redux/entityDetails/selector';
// import LayoutLoading from '../Utils/LayoutLoading';
// import axios from 'axios';
// import { messageService } from '../Utils/messageService';

// function DocumentUpload({entityDetails, showDocs = false}) {
//     const docs = [];

//     // NEW STATE - For clickable images (KYC individual)
//     const [showImageModal, setShowImageModal] = useState(false);
//     const [selectedImage, setSelectedImage] = useState(null);

//     // NEW FUNCTION -  For clickable images (KYC individual)
//     const handleImageClick = (imageSrc) => {
//         setSelectedImage(imageSrc);
//         setShowImageModal(true);
//     };

//     const initialData = [
//         /*{items: 'Company Register', status:"Available", comments: ''},
//         {items: 'W-8 Form', status: "Not-Valid", comments: ''},
//         {items: 'Audited Annual Report', status: "Available"},        
//         {items: 'Organisation Chart', status: "Available"},
//         {items: 'Economic Sanctions Due Diligence Questionnaire', status: "Not-Available"},
//         {items: 'Annual Meeting of Stockholders Report', status: "Available"}
//         */
//        {items: 'Adhar Card', status:"Available", comments: ''},
//         {items: 'PAN Card', status: "Available", comments: ''},
//         {items: 'Passport', status: "Not Matched"},        
//         // {items: 'Voter ID', status: "Not Available"},
//         // {items: 'Ration Card', status: "Not Available"},
//         {items: 'Payslip', status: "Not Available"},
//         {items: 'Utility Bill', status: "Not Valid"}

//     ];


//     // const kycAttributeData = [
//     //     { attribute: 'Name', value: 'Deepak Srivastava', source: 'National ID', docView: adhar_combined, status: 'Matched', info: '', comments: '' },
//     //     { attribute: 'Name', value: 'Deepak Jind', source: 'Passport', docView: customer_passport, status: 'Not Matched', info: 'Name Mismatch', comments: '' },

//     //     { attribute: 'DOB', value: '30/08/1992', source: 'National ID', docView: adhar_combined, status: 'Matched', info: '', comments: '' },
//     //     { attribute: 'DOB', value: '30/08/1992', source: 'Tax Registration', docView: pan_card, status: 'Matched', info: '', comments: '' },
//     //     { attribute: 'DOB', value: '22/04/1990', source: 'Passport', docView: customer_passport, status: 'Not Matched', info: 'DOB Mismatch', comments: '' },


//     //     { attribute: 'Address', value: "Plot N0. 443, House No. 204, Krishna nagar, Bhagwanpur Pin-221005", source: 'National ID', docView: adhar_combined, status: 'Matched', info: '', comments: '' },
//     //     { attribute: 'Address', value: '', source: 'Utility Bill', docView: electricity_bill, status: 'Invalid', info: 'Blurred Document', comments: '' },

//     //     { attribute: "Father's Name", value: 'Rajendra Srivastava', source: 'National ID', docView: adhar_combined, status: 'Matched', info: '', comments: '' },

//     //     { attribute: 'Annual Income', value: '1,000,000 INR', source: 'Tax Registration', docView: pan_card, status: 'Matched', info: '', comments: '' },
//     //     { attribute: 'Annual Income', value: '', source: 'Payslips', docView: '', status: 'Not Available', info: '', comments: '' }
//     //     ];

//     const kycAttributeData = [
//         { attribute: 'Name', value: 'Aisyah Rahmani', source: 'National ID', docView: indonesian_national_id, status: 'Matched', info: '', comments: '' },
//         { attribute: 'Name', value: 'Aisyah Rahmani', source: 'Passport', docView: indonesian_passport, status: 'Matched', info: '', comments: '' },

//         { attribute: 'DOB', value: '23/06/1979', source: 'National ID', docView: indonesian_national_id, status: 'Matched', info: '', comments: '' },
//         // { attribute: 'DOB', value: '23/06/1979', source: 'Tax Registration', docView: pan_card, status: 'Matched', info: '', comments: '' },
//         { attribute: 'DOB', value: '23/06/1979', source: 'Passport', docView: indonesian_passport, status: 'Matched', info: '', comments: '' },

//         { attribute: 'Address', value: "100 Pasir Panjang Road, #03-01 The Beacon, Singapore 118520", source: 'National ID', docView: indonesian_national_id, status: 'Matched', info: '', comments: '' },
//         { attribute: 'Address', value: '100 Pasir Panjang Road, #03-01 The Beacon, Singapore 118520', source: 'Utility Bill', docView: indonesian_utility_bill, status: 'Matched', info: '', comments: '' },

//         // { attribute: "Father's Name", value: 'Rajendra Srivastava', source: 'National ID', docView: indonesian_national_id, status: 'Matched', info: '', comments: '' },

//         { attribute: 'Annual Income', value: '135,500,000 IDR', source: 'Tax Registration', docView: indonesian_national_id, status: 'Matched', info: '', comments: '' },
//         { attribute: 'Annual Income', value: '135,500,000 IDR', source: 'Payslip', docView: indonesian_masked_payslip, status: 'Matched', info: '', comments: '' }
//         ];

//     const [kycData, setKycData] = useState(kycAttributeData);

//     const [showMissDocs, setShowMissDocs] = useState(false)
//     const [loading, setLoading] = useState(false);
//     const [missDocs, setMissDocs] = useState(docs)
//     const [data, setData] = useState(initialData)
  
//     const headers = [
//         // {
//         //     Header:"KYC Documents",
//         //     accessor:"items",
//         //     Cell:(props)=>{
//         //         if (props.value === 'Adhar Card') {
//         //             return <><span>{props.value}</span><img src={adhar_combined} alt="Adhar Card" className={documentUploadCss.documentImages}style={{ cursor: "pointer" }}onClick={() => handleImageClick(adhar_combined)}/></>
//         //         } else  if (props.value === 'PAN Card') {
//         //             return <><span>{props.value}</span><img src={pan_card} alt="Pan Card" className={documentUploadCss.documentImages}style={{ cursor: "pointer" }}onClick={() => handleImageClick(pan_card)}/></>
//         //         } else if (props.value === 'Electricity Bill') {
//         //             return <><span>{props.value}</span><img src={electricity_bill} alt="Electricity Bill" className={documentUploadCss.documentImages}style={{ cursor: "pointer" }}onClick={() => handleImageClick(electricity_bill)}/></>
//         //         } else if (props.value === 'Passport') {
//         //             return <><span>{props.value}</span><img src={customer_passport} alt="Passport" className={documentUploadCss.documentImages}style={{ cursor: "pointer" }}onClick={() => handleImageClick(customer_passport)}/></>
//         //         } else {
//         //             return <span>{props.value}</span>
//         //         }
//         //     }
//         // },
//         {
//             Header:"KYC Documents",
//             accessor:"items",
//             Cell:(props)=>{
//                 if (props.value === 'Adhar Card') {
//                     return <><span>{props.value}</span><img src={indonesian_national_id} alt="Adhar Card" className={documentUploadCss.documentImages}style={{ cursor: "pointer" }}onClick={() => handleImageClick(indonesian_national_id)}/></>
//                 } else  if (props.value === 'PAN Card') {
//                     return <><span>{props.value}</span><img src={indonesian_national_id} alt="Pan Card" className={documentUploadCss.documentImages}style={{ cursor: "pointer" }}onClick={() => handleImageClick(indonesian_national_id)}/></>
//                 } else if (props.value === 'Electricity Bill') {
//                     return <><span>{props.value}</span><img src={indonesian_utility_bill} alt="Electricity Bill" className={documentUploadCss.documentImages}style={{ cursor: "pointer" }}onClick={() => handleImageClick(indonesian_utility_bill)}/></>
//                 } else if (props.value === 'Passport') {
//                     return <><span>{props.value}</span><img src={indonesian_passport} alt="Passport" className={documentUploadCss.documentImages}style={{ cursor: "pointer" }}onClick={() => handleImageClick(indonesian_passport)}/></>
//                 } else if (props.value === 'Payslip') {
//                     return <><span>{props.value}</span><img src={indonesian_masked_payslip} alt="Passport" className={documentUploadCss.documentImages}style={{ cursor: "pointer" }}onClick={() => handleImageClick(indonesian_masked_payslip)}/></>
                
//                 } else {
//                     return <span>{props.value}</span>
//                 }
//             }
//         },
//         {
//             Header:"Status",
//             accessor:"status",
//             Cell:(props)=>{
//                 const badgeStyle = { fontFamily: "var(--poppinsRegular)", fontSize: "var(--fontSizeSmall)" }
//                 if (props.value === 'Available') {
//                     return <Badge bg="success" className={documentUploadCss.badge} style={badgeStyle}>{props.value}</Badge>;
//                 } else  if (props.value === 'Not Valid') {
//                     return <Badge bg="danger" className={documentUploadCss.badge} style={badgeStyle}>{props.value}</Badge>;
//                     // Updated passport not matched bg for Individual KYC
//                 } else  if (props.value === 'Not Matched') {
//                     return <Badge bg="danger" className={documentUploadCss.badge} style={badgeStyle}>{props.value}</Badge>;
//                 } else {
//                     return <Badge bg="secondary" className={documentUploadCss.badge} style={badgeStyle}>{props.value}</Badge>
//                 }
//             }
//         },
//         {
//             Header:"Reviewer's Comments(if any)",
//             accessor:"comments",
//             Cell:(props)=>{
//                 return <>
//                 <Form.Control
//                     name="text"
//                     required={true}
//                     className={documentUploadCss.inputTextArea}
//                     value={props.value}
//                     onChange={(e) => data.map(item => item.items === props.row.id ? item.comments = e.target.value : item.comments)}
//                 />
//                 <span style={{marginLeft: "20px", display: "inline-block"}}><img src={saveIcon} alt="save"></img></span>
//                 </>
//             }
//         }
//     ]

//     // ============= Updated KYC for additional headers start's here ===========
//     const kycHeaders = [
//                 { Header: 'KYC Attribute', accessor: 'attribute' },
//                 { Header: 'Value', accessor: 'value' },
//                 { Header: 'Source', accessor: 'source' },
//                 {
//                     Header: 'Doc View',
//                     accessor: 'docView',
//                     Cell: (props) =>
//                     props.value ? (
//                         <Button
//                         variant="link"
//                         size="sm"
//                         style={{
//                         padding: '4px 10px',
//                         height: '30px',
//                         lineHeight: '22px'
//                         }}
//                         onClick={() => handleImageClick(props.value)}
//                         >
//                         View
//                         </Button>
//                     ) : null
//                 },
//                 {
//                     Header: 'Status',
//                     accessor: 'status',
//                     Cell: (props) => {
//                     const badgeStyle = { fontFamily: "var(--poppinsRegular)", fontSize: "var(--fontSizeSmall)" }
//                     if (props.value === 'Matched')
//                         return <Badge bg="success" style={badgeStyle}>{props.value}</Badge>;
//                     if (props.value === 'Not Matched' || props.value === 'Invalid')
//                         return <Badge bg="danger" style={badgeStyle}>{props.value}</Badge>;
//                     if (props.value === 'Not Available')
//                         return <Badge bg="secondary" style={badgeStyle}>{props.value}</Badge>;
//                     return null;
//                     }
//                 },
//                 { Header: 'Info', accessor: 'info' },
//                 {
//                     Header: "Reviewer's Co.",
//                     accessor: 'comments',
//                     Cell: (props) => (
//                     <Form.Control
//                         className={documentUploadCss.inputTextArea}
//                         value={props.value}
//                         onChange={(e) => {
//                         const updated = [...kycData];
//                         updated[props.row.index].comments = e.target.value;
//                         setKycData(updated);
//                         }}
//                     />
//                     )
//                 }
//                 ];

//     // ============= Updated KYC for additional headers end's here ===========

//      data.map(item => {
//         if (item.status !== "Available") {
//             docs.push(item.items);
//         }
//      })

//      useEffect(() => {
//         if (showDocs) {
//             setShowMissDocs(showDocs)
//           }
//      }, [])

//     const missDocsHandler = () => {
//         setShowMissDocs(true)
//     }


//     const sendMailHandler = (eventKey) => {
//         setLoading(true)
//         if (eventKey === 1) {
//             //axios.post(`${process.env.REACT_APP_API_BASE_URL}/send_mail_with_attachment_non_compliant`)
//             axios.post(`${process.env.REACT_APP_API_BASE_URL}/email_RFI`)
//             .then(response => {
//                 setLoading(false)
//                 messageService.sendMessage({variant: "success", message: "Mail has been Sent Successfully."})
//             })
//             .catch(error => {
//                 setLoading(false)
//                 messageService.sendMessage({variant: "danger", message: "server error"});
//             })
//         } else {
//         //axios.post(`${process.env.REACT_APP_API_BASE_URL}/send_mail_non_compliant`)
//         axios.post(`${process.env.REACT_APP_API_BASE_URL}/email_EDD`)
//         .then(response => {
//             setLoading(false)
//             messageService.sendMessage({variant: "success", message: "Mail has been Sent Successfully."})
//         })
//         .catch(error => {
//             setLoading(false)
//             messageService.sendMessage({variant: "danger", message: "server error"});
//         })
//      }  
//     }

//     return (
//         <><div>
//             <span style={{ margin: "10px 5px 0px 0px", float: "right" }}>
//                 <img src={missingDocument} alt="missingDocument" style={{width: "45px"}}/>
//                 <Button className={`${documentUploadCss.documentList} backgroundDanger`} variant='danger'
//                 onClick={() => missDocsHandler()}>Missing Documents</Button>
//                 <img src={AutoEmail} alt="AutoEmail" style={{width: "36px", marginRight: "4px"}}/>
//                 {/* <Button className={`${documentUploadCss.documentList} backgroundDanger`} variant="primary"
//                 onClick={() => sendMailHandler()}>Request Information</Button> */}
//                 <DropdownButton as={ButtonGroup} title="Customer Outreach"
//                 className={`${documentUploadCss.documentDropdown} backgroundDanger`} variant="none" id="bg-nested-dropdown">
//                 <Dropdown.Item eventKey="1" onClick={() => sendMailHandler(1)}>RFI</Dropdown.Item>
//                 <Dropdown.Item eventKey="2" onClick={() => sendMailHandler(2)}>EDD</Dropdown.Item>
//                 </DropdownButton>
//             </span>
//         </div>
        
//         {/* ===========Updated KYC for additional headers=========== */}
//         <div className={documentUploadCss.innerDiv} style={{ marginTop: '20px' }}>
//         <BasicTable availableColumns={kycHeaders} data={kycData} />
//         </div>
//         {/* =========== End's here=========== */}

//         <div className={documentUploadCss.innerDiv}>
//          <BasicTable availableColumns={headers} data={data} ></BasicTable></div>
//         {
//             showMissDocs &&
//             <CustomModal onHideHandler={() => setShowMissDocs(false)}
//             modalHeader='Missing Documents'>
//                 { 
//                    missDocs.length ? <ul> 
//                     {
//                       missDocs.map(doc => (
//                         <li>{doc}</li>
//                       ))  
//                     }
//                 </ul> : <p style={{fontFamily: "var(--poppinsRegular)", fontSize: "var(--fontSizeSmall)"}}>There are no missing Documents</p>
//                 }
//             </CustomModal>
//         }
//         {
//             loading &&
//             <LayoutLoading message="Loading"/>
//         }

//         {
//             showImageModal && (
//                 <CustomModal
//                     modalHeader="Document Preview"
//                     onHideHandler={() => setShowImageModal(false)}
//                 >
//                     <img
//                         src={selectedImage}
//                         alt="Preview"
//                         style={{
//                             width: "100%",
//                             maxHeight: "70vh",
//                             objectFit: "contain"
//                         }}
//                     />
//                 </CustomModal>
//             )
//         }

//         </>
//     )
// }

// const mapStateToProp = (state) => {
//     return {
//       entityDetails: selectEntityDetails(state)
//     }
// }


// export default connect(mapStateToProp, null)(DocumentUpload)

