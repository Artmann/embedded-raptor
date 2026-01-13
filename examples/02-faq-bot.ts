/* eslint-disable no-console */
/**
 * Example: FAQ Bot
 *
 * This example shows how to build a FAQ matching system that finds
 * the most relevant question-answer pairs based on user queries.
 *
 * Use case: Customer support chatbot that automatically answers
 * common questions by finding the closest matching FAQ.
 *
 * Run: bun run examples/02-faq-bot.ts
 */

import { unlink } from 'node:fs/promises'

import { EmbeddingEngine } from '../src'

// Sample FAQ database for a SaaS product
const faqs = [
  {
    id: 'reset-password',
    question: 'How do I reset my password?',
    answer:
      'Click "Forgot Password" on the login page, enter your email, and follow the link in the reset email we send you.'
  },
  {
    id: 'cancel-subscription',
    question: 'How can I cancel my subscription?',
    answer:
      'Go to Settings > Billing > Manage Subscription, then click "Cancel Subscription". Your access continues until the end of your billing period.'
  },
  {
    id: 'data-export',
    question: 'How do I export my data?',
    answer:
      'Navigate to Settings > Data & Privacy > Export Data. Choose your format (JSON or CSV) and click "Request Export". You\'ll receive a download link via email within 24 hours.'
  },
  {
    id: 'team-invite',
    question: 'How do I invite team members?',
    answer:
      "Go to Settings > Team > Invite Members. Enter their email addresses and select their role. They'll receive an invitation email to join your workspace."
  },
  {
    id: 'payment-methods',
    question: 'What payment methods do you accept?',
    answer:
      'We accept all major credit cards (Visa, Mastercard, Amex), PayPal, and bank transfers for annual plans. Cryptocurrency payments are available on request.'
  },
  {
    id: 'data-security',
    question: 'How do you protect my data?',
    answer:
      "We use AES-256 encryption at rest and TLS 1.3 in transit. Data is backed up daily to geographically distributed servers. We're SOC 2 Type II certified and GDPR compliant."
  },
  {
    id: 'api-access',
    question: 'Do you offer API access?',
    answer:
      'Yes! API access is available on Pro and Enterprise plans. Get your API key from Settings > Developers > API Keys. Documentation is available at docs.example.com/api.'
  },
  {
    id: 'refund-policy',
    question: 'What is your refund policy?',
    answer:
      'We offer a 30-day money-back guarantee for annual plans. Monthly subscriptions can be cancelled anytime with no refund for the current month. Contact support for refund requests.'
  }
]

async function main() {
  const dbPath = './examples-faq-bot.jsonl'
  const engine = new EmbeddingEngine({ storePath: dbPath })

  console.log('🤖 FAQ Bot Example\n')
  console.log('📋 Loading FAQ database...')

  // Store all FAQ questions in batch (we search questions, return answers)
  await engine.storeMany(
    faqs.map((faq) => ({ key: faq.id, text: faq.question }))
  )

  console.log(`✓ Loaded ${faqs.length} FAQs\n`)

  // Simulate user questions (variations of actual FAQs)
  const userQuestions = [
    'I forgot my password, what should I do?',
    'Can I get my money back?',
    'How secure is my information?',
    'I want to add people to my team',
    'Is there a REST API I can use?'
  ]

  console.log('💬 User Questions:\n')

  for (const question of userQuestions) {
    console.log(`\nUser: "${question}"`)
    console.log('─'.repeat(60))

    // Search with higher threshold (0.5) for better confidence
    const results = await engine.search(question, 3, 0.5)

    if (results.length === 0) {
      console.log(
        '❌ No matching FAQ found. Consider escalating to human support.'
      )
      continue
    }

    const topMatch = results[0]
    const confidence = topMatch.similarity

    // Find the FAQ data
    const faq = faqs.find((f) => f.id === topMatch.key)

    if (!faq) continue

    // High confidence: return the answer
    if (confidence > 0.7) {
      console.log(`✅ Match found (confidence: ${confidence.toFixed(3)})`)
      console.log(`\nQ: ${faq.question}`)
      console.log(`A: ${faq.answer}`)
    }
    // Medium confidence: suggest and show alternatives
    else if (confidence > 0.5) {
      console.log(`⚠️  Possible match (confidence: ${confidence.toFixed(3)})`)
      console.log(`\nDid you mean: "${faq.question}"?`)
      console.log(`A: ${faq.answer}`)

      if (results.length > 1) {
        console.log('\nOther related FAQs:')
        for (let i = 1; i < results.length; i++) {
          const altFaq = faqs.find((f) => f.id === results[i].key)
          if (altFaq) {
            console.log(
              `  • ${altFaq.question} (${results[i].similarity.toFixed(3)})`
            )
          }
        }
      }
    }
    // Low confidence: escalate
    else {
      console.log(
        `❌ Low confidence (${confidence.toFixed(3)}) - escalating to support`
      )
    }
  }

  console.log('\n\n💡 Confidence Thresholds:')
  console.log('─'.repeat(60))
  console.log('• > 0.7: High confidence - show answer directly')
  console.log('• 0.5-0.7: Medium confidence - ask "Did you mean?"')
  console.log('• < 0.5: Low confidence - escalate to human support')

  // Cleanup
  console.log('\n🧹 Cleaning up...')
  await engine.dispose()
  await unlink(dbPath).catch(() => {})
  console.log('✓ Done!')
}

main().catch(console.error)
