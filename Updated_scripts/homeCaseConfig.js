export const HOME_CASE_CONFIG = {
  INDIA: {
    generalDetails: {
      caseInfo: {
        caseId: 'IND-2026-001',
        country: 'India',
        createdDate: '01/02/2026',
        lastUpdated: '09/02/2026',
      },
    },
    kycAttributes: [
      { attribute: 'Name', value: 'Deepak Srivastava', source: 'NATIONAL_ID', status: 'Matched', info: '', comments: '' },
      { attribute: 'Name', value: 'Deepak Jind', source: 'PASSPORT', status: 'Not Matched', info: 'Name Mismatch', comments: '' },
      { attribute: 'DOB', value: '30/08/1992', source: 'NATIONAL_ID', status: 'Matched', info: '', comments: '' },
      { attribute: 'DOB', value: '22/04/1990', source: 'PASSPORT', status: 'Not Matched', info: 'DOB Mismatch', comments: '' },
      { attribute: 'Address', value: 'Plot N0. 443, House No. 204, Krishna nagar, Bhagwanpur Pin-221005', source: 'NATIONAL_ID', status: 'Matched', info: '', comments: '' },
      { attribute: "Father's Name", value: 'Rajendra Srivastava', source: 'NATIONAL_ID', status: 'Matched', info: '', comments: '' },
      { attribute: 'Address', value: '', source: 'UTILITY',  status: 'Invalid', info: 'Blurred Document', comments: '' },
      { attribute: 'Annual Income', value: '1,000,000 INR', source: 'PAN', status: 'Matched', info: '', comments: '' },
      { attribute: 'Annual Income', value: '', source: 'PAYSLIP', status: 'Not Available', info: '', comments: '' }
    ],
    RM_comments: {
      documents: {
        missingList: ['Passport', 'Utility Bill', 'Payslips'],
        finalStatus: 'Documents listed above are not available',
      },

      review: {
        findings: ['Passport is invalid', 'Utility Bill is blurred'],
        finalStatus:
          'Kindly review the invalid/unavailable documents and resubmit.',
      },

      fulfilment: {
        updatedAttributes: ['Action required on pending documents'],
        finalStatus:
          'Kindly resubmit the documents to proceed with the case',
      },
    },
    caseSummary: {
      overview: {
        label: 'Overall Risk Assessment',
        value:
          'Customer risk assessment completed with an overall Low risk profile based on the information available at the time of assessment. CRR Completed: 09/02/2026',
      },

      kycDocumentation: {
        label: 'KYC Documentation',
        value:
          'Required KYC documentation has been obtained, verified and completed. Completion Date: 03/02/2026',
      },

      adverseMedia: {
        label: 'Adverse Media Screening',
        value:
          'Adverse media screening identified 4 hits, which were reviewed and determined to be not material. Screening Date: 04/02/2026',
      },

      pepScreeningRisk: {
        label: 'PEP Status',
        value:
          'Based on the PEP screening conducted, no Politically Exposed Person status was identified for the client. Screening Date: 04/02/2026.',
      },

      sanctionsScreeningRisk: {
        label: 'Sanctions Screening',
        value:
          'Sanctions screening indicates that the client is not listed on any applicable sanctions lists. Screening Date: 04/02/2026.',
      },

      sowDueDiligence: {
        label: 'SOW Due Diligence',
        value:
          'SOW due diligence is in progress. Identified SOW Drivers include Employment and Rental Income.',
      },
    },
    clientScreeningDetails: {
      SanctionsScreening: {
        records: [
          {
            name: 'Raden Wijaya',
            source: 'UNSC',
            hits: 0,
            sanctionResult: 'Not Matched',
            sanctionStatus: 'Low',
            reportAvailable: true,
          },
          {
            name: 'Raden Wijaya',
            source: 'FATF',
            hits: 0,
            sanctionResult: 'Not Matched',
            sanctionStatus: 'Low',
            reportAvailable: true,
          },
          {
            name: 'Raden Wijaya',
            source: 'World Check',
            hits: 0,
            sanctionResult: 'Not Matched',
            sanctionStatus: 'Low',
            reportAvailable: false,
          },
        ],
      },
      adverseMediaScreening: {
        records: [
          {
            NNSSummary:
              'During adverse media screening, an article from a reputable news source was flagged due to a name match. Comprehensive review confirmed the article pertains to a separate individual, as evidenced by discrepancies in geographic location, age, and occupation. The alert was reasonably discounted as a false positive',
            source: 'Source 1 - News',
            sourceLink: 'https://jakartapost.com',
            riskStatus: 'Low',
            sentiment: {},
            reportAvailable: true,
          },
          {
            NNSSummary:
              'Source validation and identity verification were conducted following the media alert. No corroborating evidence links the customer to the subject of the article. The customer’s risk profile remains unchanged.',
            source: 'Source 1 - Social Media',
            sourceLink: 'https://x.com',
            riskStatus: 'Low',
            sentiment: {},
            reportAvailable: true,
          },
        ],
      },
      pepScreening: {
        records: [
          {
            hits: 0,
            source: '',
            result: 'Low',
            riskStatus: {},
          },
          {
            hits: 0,
            source: '',
            result: 'Low',
            riskStatus: {},
          },
        ],
      },
    },
  },

  INDONESIA: {
    generalDetails: {
      caseInfo: {
        caseId: 'IDN-2026-001',
        country: 'Indonesia',
        createdDate: '01/02/2026',
        lastUpdated: '09/02/2026',
      },
    },
    kycAttributes: [
      { attribute: 'Name', value: 'Aisyah Rahmani', source: 'NATIONAL_ID', status: 'Matched', info: '', comments: '' },
      { attribute: 'DOB', value: '23/06/1979', source: 'PASSPORT', status: 'Matched', info: '', comments: '' },
      { attribute: 'Address', value: '100 Pasir Panjang Road, #03-01 The Beacon, Singapore 118520', source: 'UTILITY', status: 'Matched', info: '', comments: '' },
      { attribute: 'Annual Income', value: '135,500,000 IDR', source: 'PAYSLIP', status: 'Matched', info: '', comments: '' }
    ],
    RM_comments: {
      documents: {
        missingList: ['Passport', 'Utility Bill'],
        finalStatus: 'Documents listed above are not available',
      },

      review: {
        findings: ['Passport is expired', 'Utility Bill is unclear'],
        finalStatus:
          'Kindly review the invalid/unavailable documents and resubmit.',
      },

      fulfilment: {
        updatedAttributes: ['Action required on pending documents'],
        finalStatus:
          'Kindly resubmit the documents to proceed with the case',
      },
    },
    caseSummary: {
      overview: {
        label: 'Overall Risk Assessment',
        value:
          'Customer risk assessment completed with an overall Low risk profile based on the information available at the time of assessment. CRR Completed: 09/02/2026',
      },

      kycDocumentation: {
        label: 'KYC Documentation',
        value:
          'Required KYC documentation has been obtained, verified and completed. Completion Date: 03/02/2026',
      },

      adverseMedia: {
        label: 'Adverse Media Screening',
        value:
          'Adverse media screening identified 3 hits, which were reviewed and determined to be not material. Screening Date: 04/02/2026',
      },

      pepScreeningRisk: {
        label: 'PEP Status',
        value:
          'Based on the PEP screening conducted, no Politically Exposed Person status was identified for the client. Screening Date: 04/02/2026.',
      },

      sanctionsScreeningRisk: {
        label: 'Sanctions Screening',
        value:
          'Sanctions screening indicates that the client is not listed on any applicable sanctions lists. Screening Date: 04/02/2026.',
      },

      sowDueDiligence: {
        label: 'SOW Due Diligence',
        value:
          'SOW due diligence is in progress. Identified SOW Drivers include Employment and Rental Income.',
      },
    },
    clientScreeningDetails: {
      SanctionsScreening: {
        records: [
          {
            name: 'Aisyah Rahmani',
            source: 'UNSC',
            hits: 0,
            sanctionResult: 'Not Matched',
            sanctionStatus: 'Low',
            reportAvailable: true,
          },
          {
            name: 'Aisyah Rahmani',
            source: 'FATF',
            hits: 0,
            sanctionResult: 'Not Matched',
            sanctionStatus: 'Low',
            reportAvailable: true,
          },
          {
            name: 'Aisyah Rahmani',
            source: 'World Check',
            hits: 0,
            sanctionResult: 'Not Matched',
            sanctionStatus: 'Low',
            reportAvailable: false,
          },
        ],
      },
      adverseMediaScreening: {
        records: [
          {
            NNSSummary:
              'During adverse media screening, an article from a reputable news source was flagged due to a name match. Comprehensive review confirmed the article pertains to a separate individual, as evidenced by discrepancies in geographic location, age, and occupation. The alert was reasonably discounted as a false positive',
            source: 'Source 1 - News',
            sourceLink: 'https://jakartaglobe.id',
            riskStatus: 'Low',
            sentiment: {},
            reportAvailable: true,
          },
          {
            NNSSummary:
              'Source validation and identity verification were conducted following the media alert. No corroborating evidence links the customer to the subject of the article. The customer’s risk profile remains unchanged.',
            source: 'Source 1 - Social Media',
            sourceLink: 'https://x.com',
            riskStatus: 'Low',
            sentiment: {},
            reportAvailable: true,
          },
        ],
      },
      pepScreening: {
        records: [
          {
            hits: 0,
            source: 'DOW Jones',
            result: 'Low',
            riskStatus: {},
          }
        ],
      },
    },
  },
};
